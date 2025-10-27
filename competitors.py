import time

import pymysql
from dataclasses import dataclass
# from competitors_scores import AsinSimilarityCalculator
from competitors_scores import AsinSimilarityCalculator
from datetime import datetime
from ASIN_Mover_Detector import calculate_rank_changes
import logging
import math
import random
import requests
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # mysql
    # host: str = ''
    # user: str = ''
    # password: str = '.'
    host: str = ''
    user: str = ''
    password: str = ''

    # 时间窗口
    period_days: int = 7

    # 回调
    call_back_url: str = ""
    max_retries: int = 3


class KeywordsFocusProducts:
    def __init__(self, args: Config, site: str):
        self.args = args
        self.db_config = {
            'host': args.host,
            'user': args.user,
            'password': args.password,
            'database': 'amazon_asins_ca',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.site = site
        self.recommendation = AsinSimilarityCalculator(self.db_config)

    def hash_mod3(self, asin):
        """对asin哈希取模分表"""
        hash_hex = hashlib.md5(asin.encode('utf-8')).hexdigest()
        # 转为十进制整数
        hash_int = int(hash_hex, 16)
        # 对 3 取模
        return hash_int % 5

    def get_asin_search_results_data(self, ad_asin, keyword):
        """从uatu_amazon_search_results_us_general获取相关数据：分为广告位，自然位"""
        try:
            connection = pymysql.connect(**self.db_config)
            table_num = self.hash_mod3(ad_asin)
            with connection.cursor() as cursor:
                competitors_sql = f"""
                       SELECT
                           asin,
                           `rank`,
                           insert_time,
                           title,
                           price,
                           rating,
                           review_count,
                           is_sponsored,
                           src_url
                       FROM uatu_amazon_search_results_{self.site}_general_{table_num}
                       WHERE 
                           ad_asin = %s
                           AND keyword = %s
                           AND insert_time >= DATE_SUB(NOW(), INTERVAL {self.args.period_days} DAY)
                       ORDER BY asin, insert_time
                       """
                cursor.execute(competitors_sql, (ad_asin, keyword))
                competitors_data = cursor.fetchall()
                # TODO 暂时先改成取core_keyword_1不为空的，后面来修复core_keyword_1生成失败的问题
                product_sql = f"""
SELECT 
    main_info.asin,
    main_info.title,
    main_info.price,
    main_info.rating,
    main_info.review_count,
    main_info.created_at,
    main_info.core_keyword_1, main_info.core_keyword_2, main_info.core_keyword_3,
    price_position.max_price,price_position.min_price,price_position.adposition,price_position.natureposition
FROM uatu_asin_info_{self.site}_{table_num} AS main_info 
LEFT JOIN 
    (SELECT
        a.ASIN,
        a.max_price,
        a.min_price,
        d.position adposition,
        e.position natureposition
    FROM
        uatu_keyword_asin_task_{self.site}_{table_num} a
        LEFT JOIN ( SELECT asin, MAX( id ) AS id FROM uatu_keyword_search_results GROUP BY asin ) b ON a.Asin = b.asin
        LEFT JOIN uatu_keyword_items_{self.site}_{table_num} c ON b.id = c.search_result_id AND a.keyword = c.keyword
        LEFT JOIN uatu_ad_position_{self.site}_{table_num} d ON c.id = d.keyword_item_id
        LEFT JOIN uatu_rank_position_{self.site}_{table_num} e ON c.id = e.keyword_item_id
    WHERE
        a.ASIN = %s
     AND a.keyword = %s
     AND a.`status` != 2) AS price_position
ON  
    main_info.asin = price_position.ASIN
WHERE 
    main_info.asin = %s and main_info.core_keyword_1 is not null
ORDER BY main_info.created_at DESC  
LIMIT 1;
                """

                cursor.execute(product_sql, (ad_asin, keyword, ad_asin))
                product_data = cursor.fetchone()
                logging.info(f"成功从数据库获取 {len(competitors_data)} 条原始数据")

                core_keywords = [
                    product_data.get('core_keyword_1', '').strip(),
                    product_data.get('core_keyword_2', '').strip(),
                    product_data.get('core_keyword_3', '').strip()
                ]
                logging.info(f'核心关键词是：{core_keywords}')
                product_data['core_keywords'] = [kw for kw in core_keywords if kw]
                return competitors_data, product_data

        except Exception as e:
            logging.error(f"数据获得失败：{str(e)}")
            return

        finally:
            if 'connection' in locals() and connection.open:
                connection.close()

    def group_data_by_asin(self, data):
        """将数据按ASIN分组，提高后续处理效率"""
        asin_groups = {}
        for item in data:
            try:
                asin = item['asin']
                if asin not in asin_groups:
                    asin_groups[asin] = {
                        'all_data': [],
                        'sponsored_data': [],
                        'natural_data': [],
                        'latest_data': None
                    }
                asin_groups[asin]['all_data'].append(item)
                if item['is_sponsored']:
                    asin_groups[asin]['sponsored_data'].append(item)
                else:
                    asin_groups[asin]['natural_data'].append(item)

                current_latest = asin_groups[asin]['latest_data']
                if current_latest is None or item['insert_time'] > current_latest['insert_time']:
                    asin_groups[asin]['latest_data'] = item
            except Exception as e:
                logging.warning(f"{str(e)}")
                continue
        return asin_groups

    def calculate_average_rank(self, data):
        """计算平均排名"""
        if not data:
            return
        valid_ranks = [item['rank'] for item in data if item['rank'] is not None]
        if not valid_ranks:
            return None

        return math.ceil(sum(valid_ranks) / len(valid_ranks))

    def calculate_metrics_for_asins(self, ad_asin, keyword, asin_grouped_data, product_data):

        """问题：若result1或者result2 有一个赋值失败，这个asin数据怎么办（这样会出现1有，2没有）"""
        back_data = []
        save_data = []
        count = 0
        # TODO 测试使用
        # competitors_classifier = ["核心竞品", "精准竞品", "拓展竞品"]
        # advices = ["加强投放", "维持优化", "不建议投放"]

        title_similarity, overall_similarity, recommended_score, competitorClassification, competitorClassification_hk, ad_rank, natural_rank, data_advice, data_advice_hk = self.recommendation.get_similarity_and_score(
            product_data, asin_grouped_data, keyword)

        for i, (asin, data_groups) in enumerate(asin_grouped_data.items()):
            # 跳过本品
            if asin == ad_asin:
                count += 1
                continue
            try:
                # 计算广告位和自然位的平均排名
                # sponsored_avg_rank = self.calculate_average_rank(data_groups['sponsored_data'])
                # natural_avg_rank = self.calculate_average_rank(data_groups['natural_data'])
                # 获取最新的价格、评分和评论数量
                latest_data = sorted(data_groups['all_data'], key=lambda x: x['insert_time'], reverse=True)[0]

                latest_price = latest_data['price']

                latest_rating = latest_data['rating']

                latest_review_count = latest_data['review_count']
                # logging.info("latest_rating通过")

                competitor_classification = competitorClassification[i]
                # logging.info("competitor_classification通过")

                competitor_classification_hk = competitorClassification_hk[i]
                # logging.info("competitor_classification_hk通过")

                # advice = random.choice(advices)
                asin_title = data_groups['all_data'][0]['title']
                # logging.info("asin_title通过")

                src_url = data_groups['all_data'][0]['src_url']
                # logging.info("src_url通过")

                result1 = {
                    "asinCode": asin,
                    "title": asin_title,
                    "price": latest_price,
                    "rating": latest_rating,
                    "reviewCount": latest_review_count,
                    "competitorClassification": competitor_classification,
                    "competitorClassification_hk":competitor_classification_hk,
                    "recommendedScore": math.trunc(recommended_score[i] * 10) / 10,
                    "avgNaturalRank": natural_rank[i],
                    "avgSponsoredRank": ad_rank[i],
                    "src_url": src_url,
                    "advice": data_advice[i],
                    "advice_hk": data_advice_hk[i]
                }
                # logging.info(f"result1长度{len(result1)}")
                result2 = {
                    "asin1": ad_asin,
                    "title1": product_data['title'],
                    "price1": product_data['price'],
                    "rating1": product_data['rating'],
                    "review_count1": product_data['review_count'],
                    "asin2": asin,
                    "title2": asin_title,
                    "price2": latest_price,
                    "rating2": latest_rating,
                    "review_count2": latest_review_count,
                    "keyword": keyword,
                    "title_similarity": title_similarity[i],
                    "overall_similarity": overall_similarity[i],
                    "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "recommended_score": recommended_score[i],
                    "asin2_natural_rank": natural_rank[i],
                    "asin2_sponsored_rank": ad_rank[i],
                    "asin2_classification": competitor_classification,
                    "asin2_classification_hk": competitor_classification_hk,
                    "src_url": src_url,
                    "action_guideline":data_advice[i],
                    "action_guideline_hk":data_advice_hk[i]
                }
                logging.info(f"result2长度{len(result1)}")
                back_data.append(result1)
                save_data.append(result2)

            except Exception as e:
                logging.warning(f"处理ASIN {asin} 时出错: {str(e)}")
                continue
        sort_back_data = sorted(
            back_data,
            key=lambda x: x['recommendedScore'],
            reverse=True
        )
        return sort_back_data, save_data, count

    def save_to_database(self, save_data):
        """将结果保存到数据库"""
        if not save_data:
            logging.warning("没有数据需要保存")
            return
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor() as cursor:
                sql = f"""
                INSERT INTO uatu_asin_comparison_{self.site} (
                asin1, title1, price1, rating1,review_count1,
                asin2, title2, price2, rating2, review_count2,
                keyword, title_similarity, overall_similarity, comparison_date,
                recommended_score, asin2_natural_rank, asin2_sponsored_rank, AsinClassifier_name, AsinClassifier_name_hk, src_url, action_guideline, action_guideline_hk
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    title1 = VALUES(title1),
                    price1 = VALUES(price1),
                    rating1 = VALUES(rating1),
                    review_count1 = VALUES(review_count1),
                    title2 = VALUES(title2),
                    price2 = VALUES(price2),
                    rating2 = VALUES(rating2),
                    review_count2 = VALUES(review_count2),
                    title_similarity = VALUES(title_similarity),
                    overall_similarity = VALUES(overall_similarity),
                    comparison_date = VALUES(comparison_date),
                    recommended_score = VALUES(recommended_score),
                    asin2_natural_rank = VALUES(asin2_natural_rank),
                    asin2_sponsored_rank = VALUES(asin2_sponsored_rank),
                    AsinClassifier_name = VALUES(AsinClassifier_name),
                    AsinClassifier_name_hk = VALUES(AsinClassifier_name_hk),
                    src_url = VALUES(src_url),
                    action_guideline = VALUES(action_guideline),
                    action_guideline_hk = VALUES(action_guideline_hk)
                """

                batch_data = []
                for item in save_data:
                    batch_data.append((
                        item['asin1'], item['title1'], item['price1'], item['rating1'], item['review_count1'],
                        item['asin2'], item['title2'], item['price2'], item['rating2'], item['review_count2'],
                        item['keyword'], item['title_similarity'], item['overall_similarity'], item['comparison_date'],
                        item['recommended_score'], item['asin2_natural_rank'], item['asin2_sponsored_rank'],
                        item['asin2_classification'], item['asin2_classification_hk'], item['src_url'],
                        item['action_guideline'], item['action_guideline_hk']
                    ))
                # 执行批量插入或更新
                cursor.executemany(sql, batch_data)
                connection.commit()

                logging.info(f"Successfully updated {len(save_data)} records in uatu_asin_comparison table")

        except Exception as e:
            connection.rollback()
            logging.error(f"Error updating database: {str(e)}")
            raise
        finally:
            cursor.close()
            connection.close()

    def analyse(self, ad_asin, keyword):
        back_data = []
        save_data = []
        success = None
        message = None
        logging.info(f"开始处理 ASIN={ad_asin}, 关键字='{keyword}'")
        # 获取数据
        search_results_data, product_data = self.get_asin_search_results_data(ad_asin, keyword)
        logging.info("获取商品信息和历史爬虫信息成功")

        if not search_results_data or not product_data:
            success = False
            message = "没有找到相关数据"
            return success, message, back_data, save_data

        # 对数据进行分组
        logging.info("开始对数据进行分组")
        asin_grouped_data = self.group_data_by_asin(search_results_data)
        if not asin_grouped_data:
            success = False
            message = "相关数据分组失败"
            return success, message, back_data, save_data, product_data

        # 计算分析
        logging.info("开始进行back_data计算分析")
        back_data, save_data, count = self.calculate_metrics_for_asins(ad_asin, keyword, asin_grouped_data,
                                                                       product_data)

        logging.info(f"back_data:{len(back_data)}records")
        logging.info(f"save_data:{len(save_data)}records")
        logging.info(f"product_data:{count}records")

        if not back_data or not save_data:
            success = False
            message = "没有生成有效的分析结果"
            return success, message, back_data, save_data, product_data

        success = True
        message = "Success"
        return success, message, back_data, save_data, product_data

    def run(self, ad_asin, keyword):
        success, message, back_data, save_data, product_data= self.analyse(ad_asin, keyword)

        # 时间数据
        time_json_data = calculate_rank_changes(ad_asin, keyword, self.site, period_days=self.args.period_days)

        # 回调
        # self.call_back(back_data, success, message, time_json_data)

        # 保存数据库
        self.save_to_database(save_data)

        return success, message, back_data, time_json_data, product_data


# Flask
app = Flask(__name__)
site = 'ca'
app.config.from_object(Config)
config = Config()
keyword_run = KeywordsFocusProducts(config, site)


@app.route('/competitors/<string:params1>/<string:params2>', methods=['GET'])
def competitors(params1, params2):
    try:
        if params1 is None or params2 is None:
            return {
                "code": 1001,
                "message": "Invalid parameter format. Expected: ad_asin keyword",
                "data": []
            }
        ad_asin, keyword = params1, params2
        s = time.time()
        success, message, back_data, time_json_data = keyword_run.run(ad_asin, keyword)
        e = time.time()
        logging.info(f"执行时间：{e - s} s")

        return {
            "code": 200 if success else 500,
            "message": message,
            "data": {
                "competitors_data": back_data,
                "price_changes": time_json_data
            }
        }
    except Exception as e:
        app.logger.error(f"GET/POST Error: {str(e)})")
        return {
            "code": 1004,
            "message": "Internal server error",
            "data": []
        }


if __name__ == "__main__":
    # ad_asin = "B0CPHPC298"
    # keyword = "morel mushrooms"
    # config = Config()
    # keyword_run = KeywordsFocusProducts(config)
    # keyword_run.run(ad_asin, keyword)
    app.run(host='0.0.0.0', port=5000, threaded=True)
    #

