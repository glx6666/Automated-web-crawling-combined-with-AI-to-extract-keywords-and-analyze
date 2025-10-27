import re
import logging
import time
from contextlib import contextmanager
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pymysql
import json
from datetime import datetime
from bs4 import BeautifulSoup
import logging
import sys
import hashlib

# 创建文件日志
file_handler = logging.FileHandler('scraper.log', encoding='utf-8')

# 创建流日志（标准输出），并显式指定 utf-8 编码（避免 GBK 编码报错）
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[logging.FileHandler('product_scraper.log'), logging.StreamHandler()]
)

SITE_KEYWORDS = {
    'us': r'Best\sSellers\sRank',
    'uk': r'Best\sSellers\sRank',
    'fr': r'Classement\s+des\s+meilleures\s+ventes\s+d\'Amazon',
    'de': r'Amazon\s+Bestseller-Rang',
    'it': r'Posizione\s+nel\s+ranking\s+Venditori\s+su\s+Amazon',
    'es': r'Clasificación\s+en\s+los\s+más\s+vendidos\s+de\s+Amazon'
}


class ProductDetailScraper:
    def __init__(self, browser, db_config, nation):
        """
        商品详情爬虫
        :param browser: 已初始化的浏览器实例
        :param db_config: 数据库配置字典
        """
        self.browser = browser  # 新增浏览器实例引用
        #self.asin = asin

        if not browser.is_ready:
            raise RuntimeError("浏览器实例未完成初始化")
        self.driver = browser.driver
        self.main_window = self.driver.current_window_handle
        self.db_config = db_config
        self.nation = nation
        self.logger = logging.getLogger(self.__class__.__name__)

    def _log_product_data(self, data):
        """打印调试用JSON并添加日志标记"""
        try:
            # 深拷贝数据避免修改原始数据
            log_data = data.copy()

            # 添加调试元数据
            log_data["_debug"] = {
                "log_timestamp": datetime.now().isoformat(),
                "current_zip": self.browser.current_zip,  # 正确引用浏览器实例的邮编
                "page_url": self.driver.current_url
            }

            # 转换不可JSON序列化的对象
            for key, value in log_data.items():
                if isinstance(value, bytes):
                    log_data[key] = "<binary data>"
                elif isinstance(value, datetime):
                    log_data[key] = value.isoformat()

            # 生成带缩进的JSON
            json_str = json.dumps(
                log_data,
                indent=2,
                ensure_ascii=False,
                default=lambda o: str(o)
            )

            # 添加醒目标记
            debug_output = f"\n{' DEBUG JSON START ':=^80}\n" \
                           f"{json_str}\n" \
                           f"{' DEBUG JSON END ':=^80}\n"

            # 同时输出到日志和控制台
            self.logger.info(debug_output)
            print(debug_output)  # 确保控制台可见

        except Exception as e:
            self.logger.error(f"生成调试JSON失败: {str(e)}")

    def judge_asin_exist(self):
        '''
        在输入asin打开新标签页之后，如果这个asin不存在那么亚马逊会返回一张狗图片
        '''
        dog = self._safe_get_element("xpath", "//div[@id='g']/div/a/img")
        if dog is not None:
            self.logger.error("❌ ASIN不存在")
            return False
        else:
            self.logger.info("✅ ASIN存在")
            return True

    @contextmanager
    def _managed_tab(self, url):
        """上下文管理：创建和关闭独立标签页（移除邮编修改逻辑）"""
        original_windows = self.driver.window_handles

        # 使用JavaScript打开新标签页更可靠
        self.driver.execute_script("window.open('');")
        new_windows = [w for w in self.driver.window_handles if w not in original_windows]

        if not new_windows:
            self.logger.error("无法打开新标签页")
            raise RuntimeError("无法打开新标签页")

        new_window = new_windows[0]

        try:
            self.driver.switch_to.window(new_window)
            # 直接使用现有cookie访问页面
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                lambda d: d.execute_script(
                    "return document.readyState === 'complete'")
            )
            # 检测验证码
            if self.browser._is_captcha_required():
                logging.info("⚠️ 检测到验证码，开始处理...")
                # result = self.solve_captcha()
                result = self.browser.solve_captcha_new()
                status_msg = "✅ 已通过验证码" if result else "‼️ 验证码处理失败"
                logging.info(status_msg)
            else:
                logging.info(" 当前无验证码")
            # TODO 判断ASIN是否存在
            if not self.judge_asin_exist():
                raise AttributeError("Asin不存在")
            wait = WebDriverWait(self.driver, 15)
            #print("111---------------")
            # 等待产品标题加载
            title_element = wait.until(
                EC.presence_of_element_located((By.ID, "productTitle"))
            )
            #print("222---------------")
            logging.info("✅ 产品标题已加载")
            try:
                # 尝试等待常规五点描述
                bullet_points = wait.until(
                    EC.presence_of_element_located((By.ID, "feature-bullets"))
                )
                logging.info("✅ 产品五点描述已加载")
            except TimeoutException:
                try:
                    # 如果常规五点描述不存在，尝试其他可能的ID
                    bullet_points = wait.until(
                        EC.presence_of_element_located((By.ID, "featurebullets_feature_div"))
                    )
                    logging.info("✅ 产品五点描述已加载(备选ID)")
                except TimeoutException:
                    logging.warning("⚠️ 未找到产品五点描述，可能是页面结构不同或产品无此信息")
            # 添加基础验证（不涉及邮编修改）
            if "captchacharacters" in self.driver.page_source:
                raise RuntimeError("页面存在未处理验证码")
            yield
        finally:
            # 更安全的窗口关闭逻辑
            if len(self.driver.window_handles) > 1:
                try:
                    self.driver.close()
                except Exception as e:
                    logging.warning(f"关闭标签页时出现异常: {str(e)}")
            self.driver.switch_to.window(self.main_window)

    def scrape_and_save(self, asin,callback_sender=None, msg_id=None,
                        redis_conn=None, heartbeat_interval=None, last_heartbeat=None):
        """
        完整爬取流程（添加调试输出和心跳机制）
        :param asin: 商品ASIN
        :param callback_sender: 回调函数实例
        :param msg_id: 任务消息ID
        :param redis_conn: Redis连接实例
        :param heartbeat_interval: 心跳间隔时间（秒）
        :param last_heartbeat: 上次心跳时间
        :return: 是否成功, 错误信息或成功信息
        """
        scrape_situation = None

        try:
            logging.info(f"�� 尝试访问ASIN详情页： {asin}")
            if self.nation == 'us':
                state = 'com'
            elif self.nation == 'ca':
                state = 'ca'
            elif self.nation == 'uk':
                state = 'co.uk'
            elif self.nation == 'fr':
                state = 'fr'
            elif self.nation == 'de':
                state = 'de'
            elif self.nation == 'it':
                state = 'it'
            elif self.nation == 'es':
                state = 'es'
            with self._managed_tab(f"https://www.amazon.{state}/dp/{asin}"):
                # time.sleep(2)
                # 创建一个等待对象，最大等待时间为10秒
                # # TODO 在打开页面后，是否要等待必要元素，需要再考虑，上线前
                # wait = WebDriverWait(self.driver, 10)
                # # # 等待产品标题加载
                # title_element = wait.until(
                #     EC.presence_of_element_located((By.ID, "productTitle"))
                # )
                # logging.info("✅ 产品标题已加载")
                # try:
                #     # 尝试等待常规五点描述
                #     bullet_points = wait.until(
                #         EC.presence_of_element_located((By.ID, "feature-bullets"))
                #     )
                #     logging.info("✅ 产品五点描述已加载")
                # except TimeoutException:
                #     try:
                #         # 如果常规五点描述不存在，尝试其他可能的ID
                #         bullet_points = wait.until(
                #             EC.presence_of_element_located((By.ID, "featurebullets_feature_div"))
                #         )
                #         logging.info("✅ 产品五点描述已加载(备选ID)")
                #     except TimeoutException:
                #         logging.warning("⚠️ 未找到产品五点描述，可能是页面结构不同或产品无此信息")
                # self.driver.execute_script("window.stop();")
                # try:
                #     price_element = wait.until(
                #         EC.presence_of_element_located((By.ID, "priceblock_ourprice"))
                #     ) or wait.until(
                #         EC.presence_of_element_located((By.ID, "price"))
                #     ) or wait.until(
                #         EC.presence_of_element_located((By.CLASS_NAME, "a-price"))
                #     )
                #     logging.info("✅ 价格信息已加载")
                # except TimeoutException:
                #     logging.warning("⚠️ 未找到价格信息，继续处理")
                # 滚动页面以触发一些懒加载元素，但无需滚动到底部
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(2)

                # 停止加载其他资源
                # self.driver.execute_script("window.stop();")
                logging.info(f"�� 开始抓取ASIN： {asin}")

                # 初始化抓取步骤
                total_steps = 5  # 假设有5个主要步骤
                current_step = 0

                # 步骤1: 提取产品数据
                product_data = self._extract_product_data(asin)
                current_step += 1
                # self._update_heartbeat(callback_sender, msg_id, redis_conn, heartbeat_interval, last_heartbeat,
                #                        current_step, total_steps)
                # 步骤2: 日志记录产品数据
                self._log_product_data(product_data)
                current_step += 1
                # self._update_heartbeat(callback_sender, msg_id, redis_conn, heartbeat_interval, last_heartbeat,
                #                        current_step, total_steps)

                # 步骤3: 保存到数据库
                if not self._save_to_database(product_data, asin):
                    logging.info(f"TASK_FAILED, 数据库写入失败")
                    scrape_situation = f"数据库保存ASIN {asin} 数据失败"
                    self.logger.error(f"保存ASIN {asin} 数据失败")
                    return False, scrape_situation
                current_step += 1
                # self._update_heartbeat(callback_sender, msg_id, redis_conn, heartbeat_interval, last_heartbeat,
                #                        current_step, total_steps)

                # scrape_situation = f"云端数据获取成功，详情页：https://www.amazon.com/dp/{asin}"
                scrape_situation = f"https://www.amazon.{state}/dp/{asin}"
                return True, scrape_situation
        except Exception as e:
            logging.info(f"爬虫运行时错误: {str(e)}")
            scrape_situation = f"爬虫运行时错误: {str(e)}"
            if 'Asin不存在' in str(e):
                logging.info(f"Asin不存在")
                scrape_situation = f"Asin不存在"
                return "Asin_None", scrape_situation
            if 'HTTPConnectionPool' in str(e):
                logging.info(f"此代理IP出现故障,更换代理ip获取方式")
                scrape_situation = f"此代理IP出现故障,更换代理ip获取方式"
                return "HTTP", scrape_situation
            # 检查是否是代理过期导致的错误
            if "ERR_PROXY_CONNECTION_FAILED" in str(e) or "ERR_TUNNEL_CONNECTION_FAILED" in str(e):
                logging.info("爬取过程中代理过期，刷新代理后重试")
                time.sleep(2)
                return None, scrape_situation  # # 注意：这里不直接重试，而是返回None让上层函数处理重试逻辑
            return False, scrape_situation

    def _safe_get_text(self, by, selector, timeout=10):
        """安全获取元素文本，兼容 text 和 textContent"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))  # 改成 presence 更稳
            )
            text = element.text.strip()
            if not text:
                # fallback: 取 textContent，有时 text 是空的
                text = element.get_attribute("textContent").strip()
            return text
        except:
            return ""

    def _safe_get_element(self, by, selector, timeout=10):
        '''安全获取元素'''
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except:
            return None

    def _safe_get_elementslist(self, by, selector, timeout=10):
        '''安全获取元素列表'''
        try:
            # 等待至少一个元素存在
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((by, selector))
            )
            return elements
        except:
            return []

    def hash_mod3(self, asin):
        """对asin哈希取模分表"""
        hash_hex = hashlib.md5(asin.encode('utf-8')).hexdigest()
        # 转为十进制整数
        hash_int = int(hash_hex, 16)
        # 对 3 取模
        return hash_int % 5

    def _save_to_database(self, data, asin):
        """带时间判断的保存逻辑"""
        try:
            with pymysql.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    # 参数校验
                    if not isinstance(data, dict):
                        raise ValueError("数据必须为字典格式")

                    # 检查必要字段
                    required_fields = ['asin', 'title']
                    if not all(data.get(field) for field in required_fields):
                        self.logger.error("缺少必要字段ASIN或Title")
                        return False

                    table_num = self.hash_mod3(asin)

                    # 查询最新记录（使用字典游标）
                    cursor.execute(f"""
                        SELECT id, created_at 
                        FROM uatu_asin_info_{self.nation}_{table_num}
                        WHERE asin = %s 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (data['asin'],))
                    existing_record = cursor.fetchone()

                    # 将元组转换为字典
                    if existing_record:
                        column_names = [desc[0] for desc in cursor.description]
                        existing_record = dict(zip(column_names, existing_record))

                    # 时间判断逻辑
                    should_update = False
                    if existing_record:
                        cursor.execute(f"""
                            SELECT TIMESTAMPDIFF(HOUR, created_at, NOW()) AS hours_diff 
                            FROM uatu_asin_info_{self.nation}_{table_num}
                            WHERE id = %s
                        """, (existing_record['id'],))  # 使用字典键访问
                        result = cursor.fetchone()
                        if result:
                            column_names = [desc[0] for desc in cursor.description]
                            result = dict(zip(column_names, result))
                            time_diff = result['hours_diff']  # 使用字典键访问
                            should_update = time_diff < 24

                    # 数据预处理
                    processed = {
                        'asin': data['asin'],
                        'title': data['title'][:1000],
                        'price': float(data['price']) if data.get('price') else None,
                        'rating': round(float(data['rating']), 2) if data.get('rating') else None,
                        'review_count': int(data['review_count']) if data.get('review_count') else 0,
                        'src': str(data['src']) if data.get('src') else None,
                        'category_name': data.get('category_name')[:255] if data.get('category_name') else None,
                        'category_rank': (data['category_rank']) if data.get('category_rank') else None,
                        'subcategory_name': data.get('subcategory_name')[:255] if data.get(
                            'subcategory_name') else None,
                        'subcategory_rank': (data['subcategory_rank']) if data.get('subcategory_rank') else None,
                        'past_month_sales': int(data['past_month_sales']) if data.get('past_month_sales') else None,
                        'about_this_item': data.get('about_this_item')[:65535] if data.get('about_this_item') else None,
                        'product_details': data.get('product_details')[:65535] if data.get('product_details') else None,
                        'customers_say': data.get('customers_say')[:65535] if data.get('customers_say') else None,
                        'select_to_learn_more': data.get('select_to_learn_more')[:65535] if data.get(
                            'select_to_learn_more') else None
                    }

                    # 动态生成SQL
                    if should_update:
                        # 使用字典键访问existing_record
                        update_sql = f"UPDATE uatu_asin_info_{self.nation}_{table_num} SET " \
                                     "title = %s, price = %s, rating = %s, review_count = %s, src = %s, " \
                                     "category_name = %s, category_rank = %s, subcategory_name = %s, " \
                                     "subcategory_rank = %s, past_month_sales = %s, " \
                                     "about_this_item = %s, product_details = %s, " \
                                     "customers_say = %s, select_to_learn_more = %s, " \
                                     "updated_at = NOW() " \
                                     "WHERE id = %s"
                        cursor.execute(update_sql, (
                            processed['title'], processed['price'], processed['rating'], processed['review_count'],
                            processed['src'],
                            processed['category_name'], processed['category_rank'], processed['subcategory_name'],
                            processed['subcategory_rank'], processed['past_month_sales'],
                            processed['about_this_item'], processed['product_details'],
                            processed['customers_say'], processed['select_to_learn_more'],
                            existing_record['id']
                        ))
                    else:
                        insert_sql = f"INSERT INTO uatu_asin_info_{self.nation}_{table_num} (" \
                                     "asin, title, price, rating, review_count, src, " \
                                     "category_name, category_rank, subcategory_name, " \
                                     "subcategory_rank, past_month_sales, " \
                                     "about_this_item, product_details, " \
                                     "customers_say, select_to_learn_more, " \
                                     "created_at, updated_at) " \
                                     "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())"
                        cursor.execute(insert_sql, (
                            processed['asin'], processed['title'], processed['price'], processed['rating'],
                            processed['review_count'], processed['src'],
                            processed['category_name'], processed['category_rank'], processed['subcategory_name'],
                            processed['subcategory_rank'], processed['past_month_sales'],
                            processed['about_this_item'], processed['product_details'],
                            processed['customers_say'], processed['select_to_learn_more']
                        ))

                    conn.commit()
                    logging.info(f"✅ ASIN存入数据库成功 :uatu_asin_info_{self.nation}_{table_num} ")

                    return True

        except pymysql.err.ProgrammingError as e:
            self.logger.error(f"SQL语法错误: {e.args[0]}\n{e.args[1]}")
            conn.rollback()
            return False
        except KeyError as e:
            self.logger.error(f"字段访问错误: 不存在字段 {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            return False
        except pymysql.Error as e:
            self.logger.error(f"数据库错误: {e.args[0]} {e.args[1]}")
            conn.rollback()
            return False
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            return False

    def _extract_product_data(self, asin):
        """核心数据提取方法"""
        data = {"asin": asin}

        try:
            # 基础信息
            data["title"] = self._safe_get_text(By.ID, "productTitle")
            if not data["title"]:
                raise ValueError("无法提取产品标题")

            data["price"] = self._parse_price()
            #print('price',data["price"])
            data["rating"] = self._parse_rating()
            data["review_count"] = self._parse_review_count()
            data['src'] = self._src_url()

            # 类目信息
            rank_data = self._parse_category_rank()
            #print(rank_data)
            data['category_name'] = rank_data.get("main_category")
            data['category_rank'] = rank_data.get("main_rank")
            sub_categories = rank_data.get("sub_categories")
            sub_categories_rank = ','.join(str(item['rank']) for item in sub_categories)
            sub_categories_name = ','.join(item['name'] for item in sub_categories)
            data['subcategory_rank'] = sub_categories_rank
            data['subcategory_name'] = sub_categories_name

            # 销售信息
            data["past_month_sales"] = self._parse_monthly_sales()

            # 详情信息
            data["about_this_item"] = self._parse_about_item()
            data["product_details"] = self._parse_technical_details()
            print('data["product_details"]',data["product_details"])
            data["customers_say"] = self._parse_customer_voices()
            data["select_to_learn_more"] = self._parse_learn_more()
            #print('data数据为：',data)
            return data
        except Exception as e:
            self.logger.error(f"数据提取异常: {str(e)}")
            raise

    def _parse_price(self):
        """解析价格，兼容所有常见亚马逊国际站"""
        try:
            # 优先从 .a-offscreen 提取
            price_str = self._safe_get_text(By.CSS_SELECTOR, ".a-offscreen")
            print("抓到的原始价格：", price_str)
            if price_str:
                return self._convert_price(price_str)

            # 退而求其次，从整数和小数部分拼接
            whole = self._safe_get_text(By.CSS_SELECTOR, ".a-price-whole")
            fraction = self._safe_get_text(By.CSS_SELECTOR, ".a-price-fraction")
            if whole and fraction:
                full_price = whole.replace(",", "").strip() + "." + fraction.strip()
                print("抓到的原始价格：", full_price)
                return float(full_price)

            # 备用：尝试其他 ID
            fallback_ids = [
                'priceblock_ourprice',
                'priceblock_dealprice',
                'tp_price_block_total_price_ww'
            ]
            for selector in fallback_ids:
                price_str = self._safe_get_text(By.ID, selector)
                if price_str:
                    return self._convert_price(price_str)

        except Exception as e:
            self.logger.error(f"解析价格时出错: {str(e)}")

        return None

    def _convert_price(self, price_str):
        """
        将包含货币符号和本地格式的价格字符串（如 "£67.62", "67,62 €"）转为 float
        """
        price_str = price_str.strip()

        # 移除货币符号和空格
        price_str = re.sub(r"[^\d,\.]", "", price_str)

        # 欧系价格可能是 67,62（逗号为小数点）
        if "," in price_str and price_str.count(",") == 1 and "." not in price_str:
            price_str = price_str.replace(",", ".")
        elif "," in price_str and "." in price_str:
            # 美式：1,234.56，去掉千分位逗号
            price_str = price_str.replace(",", "")

        try:
            return float(price_str)
        except ValueError:
            return None

    def _parse_rating(self):
        """解析评分"""
        rating = None
        rating_str = self._safe_get_text(By.CSS_SELECTOR, "#acrPopover")
        try:
            if rating_str:
                rating_str = rating_str.replace(',', '.')
                match = re.search(r'[\d.]+', rating_str)
                if match:
                    rating = float(match.group())
        except Exception as e:
            logging.warning(f"解析评分出错: {e}")
        return rating

    def _parse_review_count(self):
        """解析评论数量"""
        count_str = self._safe_get_text(By.ID, "acrCustomerReviewText")
        if not count_str:
            count_str = self._safe_get_text(By.CSS_SELECTOR, "#reviewCountText span")
        try:
            return int(re.sub(r"\D", "", count_str))
        except:
            return 0

    def _src_url(self):
        src = self._safe_get_element(By.ID, 'landingImage')
        if src:
            src_url = src.get_attribute('src')
            return src_url
        else:
            return None

    def _parse_rank_str(self, rank_str, site="us"):
        """
        将排名字符串解析为整数，根据站点判断数字格式：
        - 美式（us, ca, jp）：1,234
        - 欧式（uk, de, fr, it, es, nl, se, pl）：1.234
        """
        if not rank_str:
            return None

        rank_str = rank_str.strip()
        european_sites = {"uk", "de", "fr", "it", "es", "nl", "se", "pl"}

        if site.lower() in european_sites:
            rank_str = rank_str.replace('.', '')
        else:
            rank_str = rank_str.replace(',', '')

        # 清除其他非数字字符
        rank_str = re.sub(r"[^\d]", "", rank_str)

        try:
            return int(rank_str)
        except ValueError:
            return None

    def _parse_category_rank(self):
        """
        解析亚马逊商品类目排名，按站点适配多语言
        """
        rank_data = {
            "main_category": '',
            "main_rank": '',
            "sub_categories": []
        }

        # 多语言正则模式集中定义
        RANK_PATTERNS_BY_SITE = {
            "us": r"#?([\d,.\s]+)\s+in\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
            "ca": r"#?([\d,.\s]+)\s+in\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
            "uk": r"#?([\d,.\s]+)\s+in\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
            "fr": r"([\d .,]+)\s+en\s+([^\n#()]+?)(?=\s*(?:\d|\n|$))",
            "de": r"Nr\.\s*([\d .,]+)\s+in\s+([^\n#()]+?)(?=\s*(?:Nr\.|\n|$))",
            "it": r"#?([\d,.\s]+)\s+in\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
            "es": r"#?([\d,.\s]+)\s+en\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
            # 默认兜底
            "default": r"#?([\d,.\s]+)\s+in\s+([^\n#()]+?)(?=\s*(?:#|\n|$))",
        }

        try:
            # ⬇️ 展开详情页（部分站点）
            try:
                expand_icon = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//div[@data-csa-c-content-id='voyager-expander-btn-t2']//a[contains(@class, 'a-expander-header') and .//span[@role='heading']]"
                    ))
                )
                expand_icon.click()
                logging.info("📂 展开详情页成功")
            except:
                logging.info("✅ 无需展开详情")

            # ⬇️ 获取详情区文本
            detail_section = self._safe_get_text(By.ID, "prodDetails") \
                             or self._safe_get_text(By.ID, "detailBulletsWrapper_feature_div") \
                             or self._safe_get_text(By.ID, "productDetails_detailBullets_sections1")

            if not detail_section:
                self.logger.warning("❌ 获取详情区失败")
                return rank_data

            detail_section = detail_section.replace('\xa0', ' ').replace('\u202f', ' ')
            site = self.nation

            # 选择适配站点的正则
            pattern = RANK_PATTERNS_BY_SITE.get(site, RANK_PATTERNS_BY_SITE["default"])
            sub_matches = re.findall(pattern, detail_section, re.IGNORECASE)

            if sub_matches:
                main_rank_str, main_cat = sub_matches[0]
                sub_categories = [
                    {"rank": self._parse_rank_str(rank.strip(), site), "name": name.strip()}
                    for rank, name in sub_matches[1:]
                ]

                rank_data = {
                    "main_rank": self._parse_rank_str(main_rank_str.strip(), site),
                    "main_category": main_cat.strip(),
                    "sub_categories": sub_categories
                }

            else:
                # 🔁 兜底匹配
                fallback_match = re.search(r'(\d+)\s+in\s+([^\n#()]+)', detail_section)
                if fallback_match:
                    rank_data = {
                        "main_rank": self._parse_rank_str(fallback_match.group(1).strip(), site),
                        "main_category": fallback_match.group(2).strip(),
                        "sub_categories": []
                    }
                else:
                    self.logger.info("⚠️ 无类目排名匹配")

            return rank_data

        except Exception as e:
            self.logger.warning(f"类目解析异常: {str(e)}")
            return rank_data


    # def _parse_category_rank(self):
    #     """完整版类目排名解析"""
    #     rank_data = {
    #         "main_category": '',
    #         "main_rank": '',  # 清理可能存在的括号
    #         "sub_categories": []
    #     }
    #     # TODO
    #     try:
    #         expand_icon = WebDriverWait(self.driver, 5).until(
    #             EC.element_to_be_clickable((By.XPATH, "//div[@data-csa-c-content-id='voyager-expander-btn-t2']//a[contains(@class, 'a-expander-header') and .//span[@role='heading' and normalize-space()='Item details']]"))
    #         )
    #         expand_icon.click()
    #         logging.info(f" 常规点击类目信息")
    #     except:
    #         logging.info(f"类目信息可以直接获取")
    #
    #     try:
    #         if self._safe_get_text(By.ID, "prodDetails"):
    #             detail_section = self._safe_get_text(By.ID, "prodDetails")
    #         elif self._safe_get_text(By.ID, "detailBulletsWrapper_feature_div"):
    #             detail_section = self._safe_get_text(By.ID, "detailBulletsWrapper_feature_div")
    #         else:
    #             logging.error('类目排名无法匹配')
    #             raise
    #         # 修正后的正则表达式模式（注意转义处理）
    #         main_pattern = r"""
    #             Best\sSellers\sRank
    #             .*?
    #             \#([\d,]+)\s+in\s+
    #             ([^#]+?)
    #             (
    #                 (?:\s*\#[\d,]+\s+in\s+[^#]+)+
    #             )
    #         """
    #         # 子类目提取模式调整（允许换行）
    #         sub_pattern = r"#([\d,]+)\s+in\s+([^\n#]+?)(?=\s*#|\s*\n|$)"
    #
    #         main_match = re.search(main_pattern, detail_section, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    #         if main_match:
    #             # print("附类目区块:", main_match.group(3))  # 调试输出
    #             sub_matches = re.findall(sub_pattern, main_match.group(3))
    #
    #             rank_data = {
    #                 "main_rank": int(main_match.group(1).replace(",", "")),
    #                 "main_category": (
    #                     lambda s: re.match(r"^([^(]+)", s).group(1).strip() if re.match(r"^([^(]+)", s) else s)(
    #                     main_match.group(2).strip()),  # 清理可能存在的括号
    #                 "sub_categories": [
    #                     {"rank": int(r.replace(",", "")), "name": n.strip()}
    #                     for r, n in sub_matches
    #                 ]
    #             }
    #         else:
    #             for line in detail_section.split('\n'):
    #                 if "Best Sellers Rank" in line:
    #                     # 同时提取两个值
    #                     match = re.search(r'#(\d+)\s+in\s+(.+)$', line)
    #                     if match:
    #                         rank_data = {
    #                             "main_rank": int(match.group(1).replace(",", "")),
    #                             "main_category": (
    #                                 lambda s: re.match(r"^([^(]+)", s).group(1).strip() if re.match(r"^([^(]+)",
    #                                                                                                 s) else s)(
    #                                 match.group(2).strip()),  # 清理可能存在的括号
    #                             "sub_categories": []
    #                         }
    #         return rank_data
    #
    #
    #     except Exception as e:
    #         self.logger.warning(f"类目解析异常: {str(e)}")
    #         rank_data = {
    #             "main_category": None,
    #             "main_rank": None,
    #             "sub_categories": []
    #         }
    #         return rank_data

    def _parse_monthly_sales(self):
        """解析月销量，多站点适配"""
        try:
            # 方法1：等待父容器+使用.text
            detail_text = self._safe_get_text(By.ID, "socialProofingAsinFaceout_feature_div")


            #print('Parent text:', element.text)  # 自动获取所有子元素文本
            print('detail_text:', detail_text)

            if not detail_text:
                return 0

            return self.convert_past_month_sales(detail_text, self.nation)

        except Exception as e:
            self.logger.warning(f"月销量解析异常: {e}")
            return 0

    @staticmethod
    def convert_past_month_sales(detail_text, nation):
        """
        提取各站点不同语言下的月销量文本中的销量数字。
        支持格式：200+、2.5k、400k+、4000+、10.000+、9 mil+ 等。
        :param detail_text: 月销量相关文本
        :param nation: 国家/站点标识（如 'us', 'uk', 'fr', 'de', 'es', 'it'）
        :return: 提取到的整数销量（默认为 0）
        """
        detail_text = detail_text.lower().replace('\xa0', ' ').strip()
        print("解析文本:", detail_text)

        # 通用提取逻辑
        def extract_number(text):
            match = re.search(r'([\d.,\s]+)\s*(k|mil)?\+?', text)
            if match:
                num_str, suffix = match.groups()
                num_str = num_str.replace(',', '').replace('.', '').replace(' ', '')
                try:
                    number = float(num_str)
                    if suffix == 'k' or suffix == 'mil':
                        number *= 1000
                    return int(number)
                except:
                    return 0
            return 0

        # 各语言关键词识别（可选增强过滤）
        keywords = {
            'us': 'bought',
            'uk': 'bought',
            'ca': 'bought',
            'fr': 'achetés',
            'de': 'gekauft',
            'es': 'comprados',
            'it': 'acquistati'
        }

        if nation in keywords:
            if keywords[nation] in detail_text:
                return extract_number(detail_text)

        # fallback，文本里虽然有数字，但关键词不匹配也尝试提取
        return extract_number(detail_text)

    def _parse_about_item(self):
        """解析'About this item'"""
        try:
            item_icon = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "nic-po-expander-heading-text"))
            )
            # 点击
            self.driver.execute_script("arguments[0].click();", item_icon)
            logging.info(f" 常规点击about_this_item")
        except:
            logging.info(f"about_this_item 可以直接获取")
        try:
            wait = WebDriverWait(self.driver, 10)
            about_this_item = self._safe_get_text(By.ID, "feature-bullets")
            if not about_this_item:
                raise NoSuchElementException("Feature bullets element found but has no text.")
            return about_this_item
        except (NoSuchElementException, TimeoutException):
            xpaths_to_try = [
                '//*[@id="productFactsDesktopExpander"]/div[1]/ul/li/span',
                '//*[@id="productFactsDesktopExpander"]/div[1]/ul/span/li/span',
                '//*[@id="productFactsDesktopExpander"]/div[1]/ul[2]/li/span'
            ]
            for xpath in xpaths_to_try:
                try:
                    elements = wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath)),
                                          message=f"Trying XPath: {xpath}")
                    about_this_item = ' '.join([elem.text.strip() for elem in elements if elem.text.strip()])
                    if about_this_item:
                        return about_this_item
                except TimeoutException:
                    continue
            return None

    def _parse_technical_details(self):
        """兼容 table 和 ul 结构的亚马逊产品详情解析"""
        details = {}

        try:

            wait = WebDriverWait(self.driver, 15)  # 最长等待15秒

            # 等待table或ul任意一个结构出现
            wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#productDetails_detailBullets_sections1")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".detail-bullet-list"))
                )
            )
            # 尝试查找 table 结构
            table_rows = self.driver.find_elements(By.CSS_SELECTOR, "#productDetails_detailBullets_sections1 tr")
            if table_rows:
                for row in table_rows:
                    try:
                        key_element = row.find_element(By.TAG_NAME, "th")
                        value_element = row.find_element(By.TAG_NAME, "td")

                        key = key_element.text.strip(": ").strip()
                        value = value_element.text.strip()
                        details[key] = value
                    except Exception as e:
                        print("解析 table 行失败：", e)
                return str(details)

            # 如果 table 没找到，尝试查找 ul 结构
            list_items = self.driver.find_elements(By.CSS_SELECTOR, ".detail-bullet-list li")
            if list_items:
                for item in list_items:
                    try:
                        spans = item.find_elements(By.TAG_NAME, "span")
                        if len(spans) >= 2:
                            key = spans[0].text.replace("‏‎", "").replace(":", "").strip()
                            value = spans[1].text.strip()
                            details[key] = value
                    except Exception as e:
                        print("解析 ul 项失败：", e)
                return str(details)

            print("未找到支持的产品详情结构。")
            return "{}"

        except Exception as e:
            print("解析产品详情失败：", e)
            return "{}"

    def _parse_customer_voices(self):
        """解析客户评价摘要"""
        try:
            wait = WebDriverWait(self.driver, 10)

            # 向下滑动到底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # 获取页面的 HTML 内容
            page_source = self.driver.page_source

            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(page_source, 'html.parser')

            # 尝试使用 data-hook 属性查找客户评价摘要
            summary = soup.find('div', attrs={'data-hook': 'cr-insights-widget-summary'})
            if summary:
                return summary.get_text(strip=True)

            # 如果上述方法失败，尝试使用 CSS 选择器
            customers_say = self._safe_get_text(By.CSS_SELECTOR, "p.a-spacing-small")
            return customers_say if customers_say else ""

        except Exception as e:
            self.logger.error(f"解析客户评价摘要时出错: {str(e)}")
            return ""

    def _parse_learn_more(self):
        """解析'Select to learn more'"""
        try:
            wait = WebDriverWait(self.driver, 10)

            # 向下滑动到底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # 获取页面的 HTML 内容
            page_source = self.driver.page_source

            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(page_source, 'html.parser')

            # 尝试使用 data-hook 属性查找选择学习更多属性
            aspects = soup.find_all(attrs={'data-hook': 'cr-insights-aspect-link'})

            # 提取所有 data-csa-c-item-id 属性
            items = [a.get('data-csa-c-item-id', '') for a in aspects]

            seen = set()
            if items:
                # 过滤空值并去重
                return ','.join([i for i in items if i and (i not in seen and not seen.add(i))])
            else:
                learn_mores = self._safe_get_elementslist(By.CSS_SELECTOR, '[data-csa-c-action="infoPopOver"]')
                if learn_mores:
                    unique_texts = []
                    for element in learn_mores:
                        text = element.text.strip()
                        if text and text not in seen:
                            seen.add(text)
                            unique_texts.append(text)
                    result_unique = ','.join(unique_texts)
                    return result_unique if result_unique else ""
                else:
                    return ""


        except Exception as e:
            self.logger.error(f"解析'Select to learn more'时出错: {str(e)}")
            return ""

    def _update_heartbeat(self, callback_sender, msg_id, redis_conn, heartbeat_interval, last_heartbeat, current_step,
                          total_steps):
        """
        更新任务状态为 still_processing 并发送回调
        :param callback_sender: 回调函数实例
        :param msg_id: 任务消息ID
        :param redis_conn: Redis连接实例
        :param heartbeat_interval: 心跳间隔时间（秒）
        :param last_heartbeat: 上次心跳时间
        :param current_step: 当前步骤
        :param total_steps: 总步骤数
        """
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            progress = int((current_step / total_steps) * 100)
            callback_sender("processing", progress, "任务进行中")
            redis_conn.hset("task_status", msg_id, "processing")
            last_heartbeat = now
        return last_heartbeat
