import argparse
import logging
import time
import re
import pymysql
import requests
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
import redis
import hashlib

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
结合卖家精灵 API
1.需要验证现在卖家精灵数据格式与之前的格式有无区别, 之前的数据内容和现在数据内容的差别，如果有新增字段，写到Gitee文档，并且备份数据库的数据，然后添加新的
2.整个过程应该是 爬详情页title/des -> 获取卖家精灵数据 ->计算保存
"""


class KeyWord:
    def __init__(self, db_config, marketplace ,model_weight=None, redis_pool=None):
        self.db_config = db_config
        self.model = SentenceTransformer(model_weight, device)
        self.searches_weight = 0.3
        self.cws_weight = 0.7
        self.pool = redis_pool

        self.seller_api = 'https://api.sellersprite.com/v1/traffic/keyword'
        self.seller_header = {
            # 'secret-key': '1ca1257a49c04d3e9635dd8444a7feb5',
            'secret-key': '0edf9b3dd9ba445196d68bf9bb5e931e',
            'content-type': 'application/json;charset=utf-8',
            'size': '5'
        }
        self.marketplace = marketplace
        self.item_list = []



    def get_asin_info(self, asin):
        """
        :return: dict
        data: 'title', 'about_this_item', 'id',
        """
        connection = pymysql.connect(**self.db_config)
        try:
            table_num = self.hash_mod3(asin)
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(
                    f"SELECT title, about_this_item FROM uatu_asin_info_{self.marketplace}_{table_num} WHERE asin = %s ORDER BY id DESC LIMIT 1",
                    (asin,))
                result_title_dec = cursor.fetchall()

                if result_title_dec:
                    # result_title_dec = pd.DataFrame(result_title_dec)
                    # data = result_title_dec[0]
                    data = result_title_dec
                    # 只要有标题存在就行
                    if data[0]['title']:
                        if not data[0]['about_this_item']:
                            data[0]['about_this_item'] = ""
                        return data, None
                    # result_tr = pd.DataFrame(result_tr)
                    # data = result_tr.assign(
                    #     title=result_title_dec['title'].iloc[0] if not result_title_dec.empty else None,
                    #     about_this_item=result_title_dec['about_this_item'].iloc[
                    #         0] if not result_title_dec.empty else None
                    # )
                    else:
                        logging.error(f"数据库中缺失ASIN {asin} 的详情页部分信息（标题）")
                        message = f"uatu_asin_info缺失ASIN {asin} 的详情页部分信息（标题）"
                        return 0, message
                else:
                    logging.error(f"数据库中无法获取ASIN {asin} 的详情页信息")
                    message = f"uatu_asin_info没有{asin}详情页信息"
                    return 0, message
        except Exception as e:
            logging.error(f"数据库中无法获取ASIN {asin} 的信息：{str(e)}")
            return 0, str(e)

        finally:
            if connection:
                connection.close()

    def fetch_seller_data(self, asin):
        try:
            payload = {
                'marketplace': self.marketplace,
                'asin': asin,
                'site': 200,
                'order':
                        {'field': 'traffic_percentage',
                         'desc': True}

            }
            response = requests.post(self.seller_api, headers=self.seller_header, json=payload)
            if response.status_code != 200:
                logging.error(f"请求失败，状态码: {response.status_code}")
                return 0, f"卖家精灵数据请求失败，状态码:{response.status_code}"
            data = response.json()
            #print(f'data:{data}')
            total = data['data'].get("total", 0)
            items = data['data']['items']
            #print(total)
            #print(len(items))
            return data, None


        except Exception as e:
            logging.error(f"获取卖家精灵数据失败: {str(e)}")
            return 0, str(e)

    # def fetch_seller_data(self, asin):
    #     try:
    #         payload = {
    #             'marketplace': self.marketplace,
    #             'asin': asin
    #         }
    #         response = requests.post(self.seller_api, headers=self.seller_header, json=payload)
    #
    #         if response.status_code == 200:
    #             logging.info(response.json())
    #             return response.json(), None
    #
    #         else:
    #             logging.error("卖家精灵数据请求失败，状态码：", response.status_code)
    #             message = f"卖家精灵数据请求失败，状态码:{response.status_code}"
    #             return 0, message
    #     except Exception as e:
    #         logging.error(f"获取卖家精灵数据失败:{str(e)}")
    #         return 0, str(e)

    def get_calculate_data(self, data, caculate_data, search_result_id, asin):
        connection = pymysql.connect(**self.db_config)
        try:
            table_num = self.hash_mod3(asin)
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = f"SELECT id FROM uatu_keyword_items_{self.marketplace}_{table_num} WHERE search_result_id = %s"
                cursor.execute(sql, (search_result_id, ))
                ids = cursor.fetchall()
                ids = pd.DataFrame(ids)
                data = pd.DataFrame(data)
                caculate_data = pd.DataFrame(caculate_data)
                total_data = caculate_data.assign(
                    title=data['title'].iloc[0] if not data.empty else None,
                    about_this_item=data['about_this_item'].iloc[
                        0] if not data.empty else None
                )
                if not ids.empty:
                    total_data['id'] = ids['id'].values
                else:
                    total_data['id'] = None
                return total_data, None
        except Exception as e:
            logging.error(f"整合计算信息失败：{str(e)}")
            message = f"整合计算信息失败：{str(e)}"
            return None, message

    def get_main_word(self, title):
        content_my = f"""
                    假设你是亚马逊的SEO，请用亚马逊搜索框的常用短句生成核心关键词，严格避免较长的词，完整句子，
                    分析用户搜索行为和长尾关键词，自动挖掘出低竞争且高流量的关键词
                    请根据亚马逊商品标题生成3个关键词，严格要求：
                    1.严格按照用户常用搜索术语，
                    2.严格满足亚马逊A9算法的搜索习惯
                    3.严格匹配产品本身特征，如颜色，形状等等,
                    4.严格分析其中的转化率与竞争度
                    5.严格避免红海类目，转为细化场景
                    6.严格由1-2个名词组成。
                    标题是：{title}
                    请以英文严格按照以下JSON格式返回结果，严格按照不要解释/分析文本：
                    {{
                    "core_ketword_1": ,
                    "core_ketword_2": ,
                    "core_ketword_3": ,
                    }}
                    """
        messages = [
            {
                "role": "user",
                "content": content_my
            }
        ]
        # client = OpenAI(
        #     api_key="sk-junzalgcymtucqihznsatwuhcyeqfueavftjtvisicaatpuh",
        #     base_url="https://api.siliconflow.cn/v1/",
        # )
        # url = "https://api.siliconflow.cn/v1/chat/completions"
        url = 'http://192.168.1.3:11434/api/generate'
        # data = {
        #     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        #     # "model": "deepseek-r1:8b",
        #     "messages": messages,
        #     "max_tokens": 1024,
        #     "temperature": 0.6,
        #     "stream": False,
        #     "top_p": 0.9
        # }
        data = {
            "model": "deepseek-r1:7b",  # 必须与ollama list显示的模型名完全一致
            "prompt": content_my,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1024,
                "seed": 42
            }
        }
        headers = {
            "Authorization": "Bearer sk-junzalgcymtucqihznsatwuhcyeqfueavftjtvisicaatpuh",
            "Content-Type": "application/json"
        }

        try:
            # response = client.chat.completions.create(
            #     model="Pro/deepseek-ai/DeepSeek-R1",
            #     messages=messages,
            #     response_format={
            #         'type': 'json_object'
            #     }
            # )
            # response = requests.post(url, json=data, headers=headers)
            response = requests.post(url, json=data)
            response.raise_for_status()
            # res = response.json()['choices'][0]['message']['content']
            res = response.json()["response"]
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, res, re.DOTALL)

            # 打印提取结果a
            match = matches[0]
            try:
                # clean_res = res.strip('```json\n').strip('```')
                # product_dict_my = json.loads(clean_res)
                product_dict_my = json.loads(match)
                return product_dict_my, None
            except Exception as e:
                logging.error(f"大模型生成核心关键词失败:{str(e)}")
                return 0, str(e)
        #
        except Exception as e:
            logging.error(f"大模型生成核心关键词失败: {str(e)}")
            return 0, str(e)

    def split_title(self, title):
        """
        标题分割
        :param title:
        :return:
        """
        # 查找第一个逗号的位置
        comma_index = title.find(',')
        if comma_index == -1:
            # 没有逗号，就强行前半部分，后半部分
            title_words = title.split(" ")
            mid_point = len(title_words) // 2
            title_first_half = " ".join(title_words[:mid_point])
            title_second_half = " ".join(title_words[mid_point:])
        else:
            # 分割标题
            title_first_half = title[:comma_index].strip()
            title_second_half = title[comma_index + 1:].strip()
        return title_first_half, title_second_half

    def remove_useless_descriptions(self, description, title):
        """
        自动去除与商品本身不相关的描述
        :param description: 分段描述嵌入(embedding)形式 [N, 384](N为具体分段数, bert 特征向量为384)
        :param title: 标题嵌入(embedding)形式 [1, 384]
        """
        # 计算标题与描述之间的相似度
        # score -> [N, 1](N段描述与标题相似度)

        scores = util.pytorch_cos_sim(description, title.unsqueeze(0))
        weights = scores.clone()
        scores = scores.squeeze().cpu().detach().numpy()
        # 采取分位数法，选取70%为上分位点，采用与标题高度相似的描述
        threshold = np.percentile(scores, 70)
        indices_greater_than_mid = np.where(scores > threshold)[0]

        # 从description中选出大于mid，并拼接，成为最终与商品本身符合的描述特征
        selected_descriptions = description[indices_greater_than_mid]

        # 通过softmax，计算其中的平滑权重, 以便后面加权融合
        # weights = torch.from_numpy(scores)
        weights = weights[indices_greater_than_mid]
        weights = torch.softmax(weights, dim=0)

        return selected_descriptions, weights

    def calculate_relevance_scores(self, asin, title, description, keywords, main_keyword, main_weight=None):
        """
        对特征进行加权池化，得到最终特征
        """
        try:
            # 获取标题前半段和后半段
            # title_words = title.split(" ")
            # mid_point = len(title_words) // 2
            # title_first_half = " ".join(title_words[:mid_point])
            # title_second_half = " ".join(title_words[mid_point:])
            if main_weight is None:
                main_weight = torch.tensor([0.6, 0.3, 0.1]).to(device)
            title_first_half, title_second_half = self.split_title(title)
            # 将长描述，分为一段段句子
            # sentences = [sent_tokenize(desc) for desc in description]
            # sentences = [sent for sublist in sentences for sent in sublist]


            # 将输入转换为嵌入向量
            # 对keyword进行加权
            embeddings_keyword = self.model.encode(keywords, convert_to_tensor=True)
            if main_keyword:
                main_keyword = list(main_keyword.values())
                embeddings_main_keyword = self.model.encode(main_keyword, convert_to_tensor=True)
                embeddings_main_keyword = torch.sum(embeddings_main_keyword * main_weight.unsqueeze(1), dim=0)
                k_score = util.pytorch_cos_sim(embeddings_keyword, embeddings_main_keyword.unsqueeze(0))
                k_weight = torch.softmax(k_score, dim=0)
                embeddings_keyword = embeddings_keyword * k_weight
            embeddings_title = self.model.encode([title, title_first_half, title_second_half],
                                                 convert_to_tensor=True)


            # 标题加权池化
            title_weights = torch.tensor([0.5, 0.35, 0.15]).to(device)  # 整体重要性最高
            embeddings_title = torch.sum(embeddings_title * title_weights.unsqueeze(1), dim=0)
            # 先初始化为0
            embeddings_description_last = torch.zeros(embeddings_title.shape[0], device=embeddings_title.device)
            if description != "" and description is not None:
                sentences = sent_tokenize(description)
                embeddings_description = self.model.encode(sentences, convert_to_tensor=True)
                # 去除与商品本身不相关的描述，并得到其中的平滑权重
                embeddings_description_last, description_weight = self.remove_useless_descriptions(
                    embeddings_description,
                    embeddings_title)
                # 描述加权池化
                embeddings_description_last = torch.sum(embeddings_description_last * description_weight, dim=0)


            # 调节标题和描述的权重， 特征融合
            product_embedding = 0.55 * embeddings_title + 0.45 * embeddings_description_last

            # 计算特征与关键词相似度
            similarity_scores = util.pytorch_cos_sim(product_embedding.unsqueeze(0),
                                                     embeddings_keyword).squeeze().cpu().detach().numpy()
            # keyword_scores = []
            # for i in range(len(keywords)):
            #     keyword_scores.append((keywords[i], float(similarity_scores[i])))
            # return dict(keyword_scores)

            return similarity_scores, None
        except Exception as e:
            logging.error(f"相关性计算失败:{str(e)}")
            message = f"相关性计算失败:{str(e)}"
            return None, message

    def batch_update_relevance(self, keyword_ids, relevance_scores, relevance_scores_avg, asin):
        table_num = self.hash_mod3(asin)
        # 创建临时表
        create_temp_sql = """
                CREATE TEMPORARY TABLE temp_correlation (
                    id INT PRIMARY KEY,
                    correlation FLOAT
                )
            """

        # 批量插入数据到临时表
        insert_temp_sql = """
                INSERT INTO temp_correlation (id, correlation)
                VALUES (%s, %s)
            """

        # 执行连表更新
        update_sql = f"""
                UPDATE uatu_keyword_items_{self.marketplace}_{table_num} AS main
                JOIN temp_correlation AS temp
                ON main.id = temp.id
                SET main.correlation = temp.correlation
            """
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # 创建临时表
                cursor.execute(create_temp_sql)

                # 准备批量数据（注意ID和得分的顺序）
                batch_data = [(int(kid), float(score)) for kid, score in zip(keyword_ids, relevance_scores)]

                # 批量插入到临时表
                cursor.executemany(insert_temp_sql, batch_data)

                # 执行更新操作
                cursor.execute(update_sql)

                avg_id = min(keyword_ids)
                # 更新平均相关性
                cursor.execute(
                    f'''
                   UPDATE uatu_keyword_items_{self.marketplace}_{table_num}
                    SET correlation_average = %s 
                    WHERE id = %s
                ''', (relevance_scores_avg, avg_id)
                )
                connection.commit()
                logging.info(f"更新相关性得分成功！")
        except Exception as e:
            logging.error(f"批量更新相关性得分时发生错误: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def batch_update_classifier(self, data, asin):
        """
        批量更新分类器字段优化方法
        :param cursor: 数据库游标
        :param data: 包含更新数据的DataFrame
        """
        table_num = self.hash_mod3(asin)
        # 创建临时表
        create_temp_sql = """
            CREATE TEMPORARY TABLE temp_classifier_update (
                id INT PRIMARY KEY,
                correlation_re INT,
                correlation_tr INT,
                correlation_name varchar(255)
            )
        """

        # 批量插入数据到临时表
        insert_sql = """
            INSERT INTO temp_classifier_update (id,correlation_re, correlation_tr, correlation_name)
            VALUES (%s, %s, %s, %s)
        """

        # 连表更新主表
        update_sql = f"""
            UPDATE uatu_keyword_items_{self.marketplace}_{table_num} AS main
            JOIN temp_classifier_update AS temp
            ON main.id = temp.id
            SET 
                main.KeywordClassifier_re = temp.correlation_re,
                main.KeywordClassifier_tr = temp.correlation_tr,
                main.KeywordClassifier_name = temp.correlation_name
        """

        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # 创建临时表
                cursor.execute(create_temp_sql)

                # 准备批量数据（id, re, tr）
                batch_data = [
                    (int(row['id']),
                     int(row['KeywordClassifier_re']),
                     int(row['KeywordClassifier_tr']),
                     str(row['KeywordClassifier_name'])
                     )
                    for _, row in data.iterrows()
                ]

                # 批量插入到临时表
                cursor.executemany(insert_sql, batch_data)

                # 执行连表更新
                cursor.execute(update_sql)
                connection.commit()
                logging.info("更新相关性, 流量性成功")
        except Exception as e:
            logging.error(f"批量更新相关性, 流量性失败：{str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def update_seller_data(self, cursor, data, asin):

        total = data.get('total', 0)
        response_asin = data.get('asin', asin)
        updated_time = data.get('updatedTime', 0)
        try:
            table_num = self.hash_mod3(asin)
            # 更新uatu_keyword_search_results表
            cursor.execute(f'''
                                  INSERT INTO uatu_amazon_search_results_{self.marketplace}_general_{table_num} (marketplace, asin, total, updated_time)
                                  VALUES (%s, %s, %s, %s)
                              ''', (self.marketplace, response_asin, total, updated_time))

            search_result_id = cursor.lastrowid

            # 更新uatu_keyword_items
            for item in data.get('items', []):
                keyword = item.get('keyword', '')

                cursor.execute(f'''
                           INSERT INTO uatu_keyword_items_{self.marketplace}_{table_num} (
                               search_result_id, keyword, keyword_cn, searches, products, purchases, purchase_rate, bid, 
                               bid_max, bid_min, searches_rank, searches_rank_time_from, searches_rank_time_to, 
                               latest_1days_ads, latest_7days_ads, latest_30days_ads, supply_demand_ratio, traffic_percentage, 
                               traffic_keyword_type, conversion_keyword_type, calculated_weekly_searches
                           ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ''', (
                    search_result_id, keyword, item.get('keywordCn', ''), item.get('searches', 0),
                    item.get('products', 0), item.get('purchases', 0), item.get('purchaseRate', 0), item.get('bid', 0),
                    item.get('bidMax', 0), item.get('bidMin', 0), item.get('searchesRank', 0),
                    item.get('searchesRankTimeFrom', 0),
                    item.get('searchesRankTimeTo', 0), item.get('latest1daysAds', 0), item.get('latest7daysAds', 0),
                    item.get('latest30daysAds', 0), item.get('supplyDemandRatio', 0), item.get('trafficPercentage', 0),
                    item.get('trafficKeywordType', ''), item.get('conversionKeywordType', ''),
                    item.get('calculatedWeeklySearches', 0),
                ))
                keyword_item_id = cursor.lastrowid

                for badge in item.get('badges', []):
                    cursor.execute(f'''
                               INSERT INTO uatu_keyword_badges_{self.marketplace}_{table_num} (keyword_item_id, badge)
                               VALUES (%s, %s)
                           ''', (keyword_item_id, badge))

                rank_position = item.get('rankPosition', {})
                if rank_position:
                    cursor.execute(f'''
                               INSERT INTO uatu_rank_position_{self.marketplace}_{table_num} (
                                   keyword_item_id, page, page_size, index_position, position, updated_time
                               ) VALUES (%s, %s, %s, %s, %s, %s)
                           ''', (
                        keyword_item_id, rank_position.get('page', 0), rank_position.get('pageSize', 0),
                        rank_position.get('index', 0), rank_position.get('position', 0),
                        rank_position.get('updatedTime', 0)
                    ))

                ad_position = item.get('adPosition', {})
                if ad_position:
                    cursor.execute(f'''
                               INSERT INTO uatu_ad_position_{self.marketplace}_{table_num} (
                                   keyword_item_id, page, page_size, index_position, position, updated_time
                               ) VALUES (%s, %s, %s, %s, %s, %s)
                           ''', (
                        keyword_item_id, ad_position.get('page', 0), ad_position.get('pageSize', 0),
                        ad_position.get('index', 0), ad_position.get('position', 0), ad_position.get('updatedTime', 0)
                    ))
            print(f'api保存 {asin} 信息成功')
            return 1
        except Exception as e:
            print("api_asin详情插入失败：", str(e))
            return 0

    def hash_mod3(self, asin):
        """对asin哈希取模分表"""
        hash_hex = hashlib.md5(asin.encode('utf-8')).hexdigest()
        # 转为十进制整数
        hash_int = int(hash_hex, 16)
        # 对 3 取模
        return hash_int % 5

    def batch_update_seller_data(self, data, asin):
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor() as cursor:
                total = data.get('total', 0)
                print('total', total)
                response_asin = data.get('asin', asin)
                updated_time = data.get('updatedTime', 0)
                items = data.get('items',[])
                print('item', len(items))
                table_num = self.hash_mod3(asin)
                # 1. 使用单条SQL插入uatu_keyword_search_results表
                cursor.execute(f'''
                    INSERT INTO uatu_keyword_search_results_{self.marketplace}_{table_num} (marketplace, asin, total, updated_time)
                    VALUES (%s, %s, %s, %s)
                ''', (self.marketplace, response_asin, total, updated_time))

                search_result_id = cursor.lastrowid

                # 2. 批量准备数据
                keyword_items_data = []
                badges_data = []
                rank_position_data = []
                ad_position_data = []
                caculate_data = []
                # 3. 只循环一次，收集所有表的数据
                for item in items:
                    keyword = item.get('keyword', '')
                    caculate_data.append(
                        {
                            "keyword": keyword,
                            "searches": item.get('searches', 0) or 0,
                            "calculated_weekly_searches":item.get('calculatedWeeklySearches', 0) or 0,
                        }
                    )
                    # 为uatu_keyword_items准备数据
                    keyword_items_data.append((
                        search_result_id, keyword, item.get('keywordCn', ''), item.get('searches') or 0,
                        item.get('products') or 0, item.get('purchases') or 0, item.get('purchaseRate') or 0,
                        item.get('bid') or 0, item.get('bidMax') or 0, item.get('bidMin') or 0,
                        item.get('searchesRank') or 0, item.get('searchesRankTimeFrom') or 0,
                        item.get('searchesRankTimeTo') or 0, item.get('latest1daysAds') or 0,
                        item.get('latest7daysAds') or 0, item.get('latest30daysAds') or 0,
                        item.get('supplyDemandRatio') or 0, item.get('trafficPercentage') or 0,
                        item.get('trafficKeywordType', ''), item.get('conversionKeywordType', ''),
                        item.get('calculatedWeeklySearches') or 0, item.get('titleDensity') or 0,
                        item.get('spr') or 0, item.get('monopolyClickRate') or 0, item.get('top3ClickingRate') or 0,
                        item.get('top3ConversionRate') or 0, item.get('clicks') or 0, item.get('impressions') or 0,
                        item.get('topAsins', ''), item.get('searchesTrend', '')
                    ))

                    # 获取当前插入项的ID (需要使用last_insert_id变量记录位置)
                    current_index = len(keyword_items_data) - 1

                    # 为badges准备数据
                    for badge in item.get('badges', []):
                        badges_data.append((current_index, badge))

                    # 为rank_position准备数据
                    rank_position = item.get('rankPosition', {})
                    if rank_position:
                        rank_position_data.append((
                            current_index,
                            rank_position.get('page', 0),
                            rank_position.get('pageSize', 0),
                            rank_position.get('index', 0),
                            rank_position.get('position', 0),
                            rank_position.get('updatedTime', 0)
                        ))

                    # 为ad_position准备数据
                    ad_position = item.get('adPosition', {})
                    if ad_position:
                        ad_position_data.append((
                            current_index,
                            ad_position.get('page', 0),
                            ad_position.get('pageSize', 0),
                            ad_position.get('index', 0),
                            ad_position.get('position', 0),
                            ad_position.get('updatedTime', 0)
                        ))

                # 4. 批量插入uatu_keyword_items
                if keyword_items_data:
                    cursor.executemany(f'''
                        INSERT INTO uatu_keyword_items_{self.marketplace}_{table_num} (
                            search_result_id, keyword, keyword_cn, searches, products, purchases, purchase_rate, bid, 
                            bid_max, bid_min, searches_rank, searches_rank_time_from, searches_rank_time_to, 
                            latest_1days_ads, latest_7days_ads, latest_30days_ads, supply_demand_ratio, traffic_percentage, 
                            traffic_keyword_type, conversion_keyword_type, calculated_weekly_searches, title_density, spr, monopoly_click_rate,
                            top3_clicking_rate, top3_conversion_rate, clicks, impressions, top_asins, searchesTrend
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s)
                    ''', keyword_items_data)

                    # 获取第一个插入项的ID
                    first_item_id = cursor.lastrowid

                    # 5. 更新其他表的关联ID
                    if badges_data:
                        real_badges_data = [(first_item_id + idx, badge) for idx, badge in badges_data]
                        cursor.executemany(f'''
                            INSERT INTO uatu_keyword_badges_{self.marketplace}_{table_num} (keyword_item_id, badge)
                            VALUES (%s, %s)
                        ''', real_badges_data)

                    if rank_position_data:
                        real_rank_position_data = [
                            (first_item_id + idx, page, page_size, index_pos, position, updated_time)
                            for idx, page, page_size, index_pos, position, updated_time in
                            rank_position_data]
                        cursor.executemany(f'''
                            INSERT INTO uatu_rank_position_{self.marketplace}_{table_num} (
                                keyword_item_id, page, page_size, index_position, position, updated_time
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                        ''', real_rank_position_data)

                    if ad_position_data:
                        real_ad_position_data = [
                            (first_item_id + idx, page, page_size, index_pos, position, updated_time)
                            for idx, page, page_size, index_pos, position, updated_time in
                            ad_position_data]
                        cursor.executemany(f'''
                            INSERT INTO uatu_ad_position_{self.marketplace}_{table_num} (
                                keyword_item_id, page, page_size, index_position, position, updated_time
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                        ''', real_ad_position_data)

                logging.info(f'卖家精灵api保存 {asin} 信息成功')
                connection.commit()
                return 1, (caculate_data, search_result_id)
        except Exception as e:
            logging.error(f"api_asin详情插入失败:{str(e)}")
            message = f"更新卖家精灵数据，api_asin详情插入失败:{str(e)}"
            return 0, message
        finally:
            if connection:
                connection.close()

    def clean_data(self, data):
        """
        清洗数据，移除包含5个或更多0值的行

        参数:
        data (pd.DataFrame): 原始数据

        返回:
        pd.DataFrame: 清洗后的数据
        """
        zero_count = (data == 0).sum(axis=1)  # 计算每行中0值的数量
        cleaned_data = data[zero_count < 5]  # 筛选出0值少于5的行
        print("Available columns in the data after cleaning:", cleaned_data.columns)  # 调试：检查清洗后的数据的列名
        return cleaned_data

    def handle_missing_values(self, data):
        """
        处理缺失值，将 NaN 填充为特征的均值

        参数:
        data (pd.DataFrame): 包含缺失值的DataFrame

        返回:
        pd.DataFrame: 填充后的数据
        """
        data = data.fillna(data.mean())
        return data

    def process_kmeans(self, data_search, search_weight=0.3, cws_weight=0.7):
        """
        对流量数据进行预处理
        流量数据差异过大，比较失效，加权后特征不明显
        """
        # data_search[data_search == 0] = 1e-6
        # features = np.log1p(data_search)

        # 初始化归一化设置
        scaler = MinMaxScaler()
        # 区分0值处理
        # 仅处理总搜索量
        data_search[:, 0] = np.where(data_search[:, 0] == 0, 1e-6, data_search[:, 0])
        # 保留周搜索零值
        data_search[:, 1] = np.where(data_search[:, 1] == 0, 0, data_search[:, 1])
        features = np.empty_like(data_search)
        # 总搜索取log
        features[:, 0] = np.log1p(data_search[:, 0])
        # 周搜索取平方根压缩
        features[:, 1] = np.sqrt(data_search[:, 1])
        # 归一化
        features = scaler.fit_transform(features)
        # 特征加权
        weight = np.array([search_weight, cws_weight])
        features = features * weight

        return features

    def initial_classification(self, data, correlation):
        """
                使用 correlation 特征进行初步相关性分类
        - 高相关性：correlation > 上四分位 标签为1
        - 低相关性：correlation ： 中间部分 标签为0
        - 完全不相关：correlation < 下四分位 标签为-1

        参数:
        data (pd.DataFrame): 包含特征的DataFrame

        返回:
        pd.DataFrame: 添加初步分类结果的数据
        """
        try:
            # data = self.handle_missing_values(data)  # 处理缺失值
            # correlation_data = data['correlation']
            # median_correlation = correlation_data.median()
            # data['KeywordClassifier_re'] = (correlation_data > median_correlation).astype(int)
            # 取出相关性数据，并转为numpy加速运算
            # correlation_data = np.array(correlation, dtype=float)
            correlation_data = correlation
            # TODO
            # 采用分位数法，暂定取前75%为高相关性
            threshold_high = np.percentile(correlation_data, 85)
            threshold_low = np.percentile(correlation_data, 25)
            indices_greater_than_high = np.where(correlation_data > threshold_high)[0]
            indices_lower_than_low = np.where(correlation_data < threshold_low)[0]
            re_np = np.zeros(len(correlation))
            re_np[indices_greater_than_high] = 1
            re_np[indices_lower_than_low] = -1
            data['KeywordClassifier_re'] = re_np
            return data, None
        except Exception as e:
            logging.error(f"相关性分类失败:{str(e)}")
            message = f"相关性分类失败:{str(e)}"
            return 0, message

    def searches_classification(self, data):
        """
        针对关键词本身进行分类，异常高的值以及在正常值中倒数排序后前x个加起来能达到正常值总和的数
        0为低流量 ，1为极端高流量（二级类目），2为非异常词中的高流量
        """

        try:
            searches = np.array(data['searches'])
            # --- 第一步：用IQR法识别异常值 ---
            Q1 = np.percentile(searches, 25)
            Q3 = np.percentile(searches, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 2 * IQR

            # 标记所有异常值（x < lower 或 x > upper）
            outliers_mask = (searches < lower) | (searches > upper)

            # --- 第二步：标记 x > upper 的异常值为1 ---
            marking = np.zeros(len(searches), dtype=int)  # 初始化标记数组
            marking[searches > upper] = 1  # 标记为1

            # --- 第三步：处理非异常值，计算前70%和的标记 ---
            clean_data = searches[~outliers_mask]  # 剔除所有异常值后的数据

            # 降序排序并记录原始索引（相对于clean_data）
            sorted_indices = np.argsort(-clean_data)  # 降序排列的索引
            sorted_values = clean_data[sorted_indices]

            # 计算累积和及目标阈值
            cumulative_sum = np.cumsum(sorted_values)
            total = cumulative_sum[-1] if len(cumulative_sum) > 0 else 0  # 避免空数据
            target = 0.7 * total

            # 找到达到目标的最小索引
            if total == 0:
                n = 0  # 处理全零数据
            else:
                n = np.argmax(cumulative_sum >= target) + 1  # 计算前n项

            # 获取这些值在原始数据中的索引（仅非异常部分）
            original_clean_indices = np.where(~outliers_mask)[0]  # 非异常值的原始索引
            selected_clean_indices = sorted_indices[:n]  # clean_data中需要标记的索引
            selected_original_indices = original_clean_indices[selected_clean_indices]  # 映射回原始索引

            # 在标记数组中标记为2
            marking[selected_original_indices] = 2
            data['searches_cs'] = marking
            return data, None
        except Exception as e:
            logging.error(f"关键词流量性分类失败:{str(e)}")
            message = f"关键词流量性分类失败:{str(e)}"
            return 0, message

    def secondary_classification(self, data):
        """
        进行高低流量的分类
        - 高流量：关键词本身有显著高的月搜索量或者关键词的月搜索量在正常关键词中能排进前70%而且本品的曝光获得率能达到10%
        - 低流量：剩下的所有词

        参数:
        data (pd.DataFrame): 包含初步分类结果的数据

        返回:
        pd.DataFrame: 添加最终分类结果的数据
        """

        try:
            index_high = np.where((data['calculated_weekly_searches'] * 4 > data['searches'] * 0.1) & (data['searches_cs'] == 2))[0]
            index_corehigh = np.where(data['searches_cs'] == 1)[0]
            # index_low = np.where((data['calculated_weekly_searches']*4 <= data['searches']*0.1))[0]
            re_np = np.zeros(len(data['searches']))
            re_np[index_high] = 1
            re_np[index_corehigh] = 1
            # re_np[index_low] = 0
            data['KeywordClassifier_tr'] = re_np
            # 取出数据，并转为numpy加速运算
            # data_search = data[['searches', 'calculated_weekly_searches']].to_numpy(dtype=float)
            # # 特征预处理
            # features_search = self.process_kmeans(data_search, self.searches_weight, self.cws_weight)
            # # 初始化KMeans设置，分为高中低三类流量，中低流量一同视为低流量
            # kmeans = KMeans(n_clusters=3, random_state=42)
            # # 聚类
            # kmeans.fit(features_search)
            # # 获取聚类标签
            # cluster_labels = kmeans.labels_
            # # 获取簇中心
            # cluster_centers = kmeans.cluster_centers_
            #
            # tr_weight = np.array([self.searches_weight, self.cws_weight])
            # # 特征加权后，计算三类簇中心特征值，选出综合最大的为高流量
            # tr_score = np.dot(cluster_centers, tr_weight)
            # max_index = np.argmax(tr_score)
            # tr_np = (cluster_labels == max_index).astype(int)
            # data['KeywordClassifier_tr'] = tr_np
            return data, None
        except Exception as e:
            logging.error(f"流量性分类失败:{str(e)}")
            message = f"流量性分类失败:{str(e)}"
            return 0, message

    def keywords_classification(self, data):
        '''
        关键词分类名称
        KeywordClassifier_re为1，0，-1分别对应高，中，低相关性
        KeywordClassifier_tr为1，0分别对应高，低流量
        共有3×2种分类分别对应在classfication
        '''
        data_search = data[['KeywordClassifier_re', 'KeywordClassifier_tr']]
        classfication = ['核心热词', '精准长尾', '拓展热词', '潜力词', '泛流量词', '边缘词']
        conditions = [
            (data_search['KeywordClassifier_re'] == 1) & (data_search['KeywordClassifier_tr'] == 1),
            (data_search['KeywordClassifier_re'] == 1) & (data_search['KeywordClassifier_tr'] == 0),
            (data_search['KeywordClassifier_re'] == 0) & (data_search['KeywordClassifier_tr'] == 1),
            (data_search['KeywordClassifier_re'] == 0) & (data_search['KeywordClassifier_tr'] == 0),
            (data_search['KeywordClassifier_re'] == -1) & (data_search['KeywordClassifier_tr'] == 1),
            (data_search['KeywordClassifier_re'] == -1) & (data_search['KeywordClassifier_tr'] == 0),
        ]

        data['KeywordClassifier_name'] = np.select(conditions, classfication, default='未知分类')

        return data

    def classification(self, data, correlation):
        """
        使用 correlation 特征进行初步相关性分类
            - 高相关性：correlation > 上四分位 标签为1
            - 低相关性：correlation ： 中间部分 标签为0
            - 完全不相关：correlation < 下四分位 标签为-1

        进行高低流量的分类
        - 高流量：关键词本身有显著高的月搜索量或者关键词的月搜索量在正常关键词中能排进前70%而且本品的曝光获得率能达到10%
        - 低流量：剩下的所有词

         关键词分类名称
        KeywordClassifier_re为1，0，-1分别对应高，中，低相关性
        KeywordClassifier_tr为1，0分别对应高，低流量
        共有3×2种分类分别对应在classfication
        参数:
            data (pd.DataFrame): 包含特征的DataFrame
    """
        try:
            # TODO 1 流量性分类
            data, _ = self.searches_classification(data)
            data, _ = self.secondary_classification(data)

            correlation_data = correlation
            # TODO 2 相关性分类
            # 采用分位数法，暂定取前75%为高相关性
            threshold_high = np.percentile(correlation_data, 85)
            threshold_low = np.percentile(correlation_data, 25)
            indices_greater_than_high = np.where(correlation_data > threshold_high)[0]
            indices_lower_than_low = np.where(correlation_data < threshold_low)[0]
            re_np = np.zeros(len(correlation))
            re_np[indices_greater_than_high] = 1
            re_np[indices_lower_than_low] = -1
            data['KeywordClassifier_re'] = re_np

            # TODO 3 关键词分类名称
            data_search = data[['KeywordClassifier_re', 'KeywordClassifier_tr']]
            # TODO 现关键词分类依据不足，容易造成分类混淆
            # 如：假如所有关键词的所有词根数平均为6个，在长尾词中出现了一个词根为2。这样是不符合逻辑。除了计算词根数时，也需要考虑实际曝光量大小
            classfication = ['核心热词', '精准长尾', '拓展热词', '潜力词', '泛流量词', '边缘词']
            classfication_hk = ['核心熱詞', '精準長尾', '拓展熱詞', '潛力詞', '泛流量詞', '邊緣詞']
            conditions = [
                (data_search['KeywordClassifier_re'] == 1) & (data_search['KeywordClassifier_tr'] == 1),
                (data_search['KeywordClassifier_re'] == 1) & (data_search['KeywordClassifier_tr'] == 0),
                (data_search['KeywordClassifier_re'] == 0) & (data_search['KeywordClassifier_tr'] == 1),
                (data_search['KeywordClassifier_re'] == 0) & (data_search['KeywordClassifier_tr'] == 0),
                (data_search['KeywordClassifier_re'] == -1) & (data_search['KeywordClassifier_tr'] == 1),
                (data_search['KeywordClassifier_re'] == -1) & (data_search['KeywordClassifier_tr'] == 0),
            ]

            data['KeywordClassifier_name'] = np.select(conditions, classfication, default='未知分类')
            data['KeywordClassifier_name_hk'] = np.select(conditions, classfication_hk, default='未知分类')

            return data, None

        except Exception as e:
            logging.error(f"关键词分类错误：{str(e)}")
            message = f"关键词分类错误：{str(e)}"
            return None, message

    def save_llm_mainword(self, asin, mainword):

        connection = pymysql.connect(**self.db_config)
        try:
            table_num = self.hash_mod3(asin)
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:

                if isinstance(mainword, dict):
                    # 字典类型处理
                    core1 = mainword.get("core_ketword_1")
                    core2 = mainword.get("core_ketword_2")
                    core3 = mainword.get("core_ketword_2")
                elif isinstance(mainword, (set, list)):
                    # 集合/列表类型处理（注意集合的无序性）
                    keywords = list(mainword)
                    # 按顺序填充并补None到3个元素
                    core1 = keywords[0] if len(keywords) > 0 else None
                    core2 = keywords[1] if len(keywords) > 1 else None
                    core3 = keywords[2] if len(keywords) > 2 else None
                else:
                    raise ValueError("Unsupported keywords format")
                params = (core1, core2, core3, asin)
                update_sql = f"""
                            UPDATE uatu_asin_info_{self.marketplace}_{table_num} 
                            SET core_keyword_1 = %s, core_keyword_2 = %s, core_keyword_3 = %s 
                            WHERE asin = %s
                         """
                cursor.execute(update_sql, params)
                logging.info("成功更新大模型核心词！！！")
                connection.commit()
                return -1, "Success"
        except Exception as e:
            logging.error(f"直接更新大模型失败:{str(e)}")
            message = f"直接更新大模型失败:{str(e)}"
            return 0, message
        finally:
            if connection:
                connection.close()

    def batch_update_data(self, data, relevance_scores_avg, avg_id, asin):
        # 创建临时表
        create_temp_sql = """
            CREATE TEMPORARY TABLE temp_classifier_update (
                id INT PRIMARY KEY,
                correlation FLOAT ,
                correlation_re INT,
                correlation_tr INT,
                correlation_name varchar(255),
                correlation_name_hk varchar(225),
                action_guideline varchar(255),
                action_guideline_hk varchar(225)
            )
        """

        # 批量插入数据到临时表
        insert_sql = """
                    INSERT INTO temp_classifier_update (id,correlation, correlation_re, correlation_tr, correlation_name, correlation_name_hk, action_guideline, action_guideline_hk)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

        # 连表更新主表
        table_num = self.hash_mod3(asin)
        update_sql = f"""
                   UPDATE uatu_keyword_items_{self.marketplace}_{table_num} AS main
                   JOIN temp_classifier_update AS temp
                   ON main.id = temp.id
                   SET
                       main.correlation = temp.correlation,
                       main.KeywordClassifier_re = temp.correlation_re,
                       main.KeywordClassifier_tr = temp.correlation_tr,
                       main.KeywordClassifier_name = temp.correlation_name,
                       main.KeywordClassifier_name_hk = temp.correlation_name_hk,
                       main.action_guideline = temp.action_guideline,
                       main.action_guideline_hk = temp.action_guideline_hk
               """
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # 创建临时表
                cursor.execute(create_temp_sql)
                # 准备批量数据（id, re, tr）
                batch_data = [
                    (int(row['id']),
                     float(row['correlation']),
                     int(row['KeywordClassifier_re']),
                     int(row['KeywordClassifier_tr']),
                     str(row['KeywordClassifier_name']),
                     str(row['KeywordClassifier_name_hk']),
                     str(row['Keyword_advice']),
                     str(row['Keyword_advice_hk'])
                     )
                    for _, row in data.iterrows()
                ]
                # 批量插入到临时表
                cursor.executemany(insert_sql, batch_data)
                # 执行连表更新
                cursor.execute(update_sql)
                # 更新平均相关性
                cursor.execute(
                    f'''
                       UPDATE uatu_keyword_items_{self.marketplace}_{table_num}
                        SET correlation_average = %s 
                        WHERE id = %s
                    ''', (relevance_scores_avg, avg_id)
                )
                connection.commit()
                logging.info("更新相关性, 流量性成功")
                return 1, None
        except Exception as e:
            logging.error(f"批量更新相关性, 流量性失败：{str(e)}")
            message = f"批量更新相关性, 流量性失败：{str(e)}"
            return 0, message
        finally:
            if connection:
                connection.close()

    def keyword_advice_classification(self, data):
        '''
        :param data: data数据框
        :return:

        核心热词（高相关性-高流量）
        ┣ 曝光获得率 >30%：拓展ASIN投放 + 防御性ASIN否定
        ┣ 曝光获得率15%-30%：保持关键词竞价 + 动态监测转化率
        ┣ 曝光获得率5%-15%：提升关键词竞价 + 精准ASIN投放
        ┗ 曝光获得率 <5%：全面加强关键词投放 + 精准ASIN投放 + 检查Listing相关性

        精准长尾（高相关性-低流量）
        ┣ 曝光获得率 >60%：优化转化率 > 拓展流量（因流量天花板低）
        ┣ 曝光获得率30%-60%：测试长尾词变体投放（包含单复数/错别字）
        ┗ 曝光获得率 <30%：提升长尾词竞价 + 拓展同义词库

        拓展热词（中相关性-高流量）
        ┣ 曝光获得率 >20%：启用ASIN定向收割 + 短语匹配
        ┣ 曝光获得率10%-20%：采用动态竞价（仅降低） + 精准匹配
        ┗ 曝光获得率 <10%：选择性投放（需配合转化率数据判断）

        潜力词（中相关性-低流量）
        ┣ 曝光获得率 >50%：选择性埋词优化提升相关性
        ┣ 曝光获得率20%-50%：梯度竞价测试（寻找最佳ROAS点）
        ┗ 曝光获得率 <20%：暂停投放（需先优化产品属性匹配度）

        泛流量词（低相关性-高流量）
        ┣ 曝光获得率 >10%：立即设置否定关键词
        ┣ 曝光获得率5%-10%：观察转化率（高于平均ACOS则否定）
        ┗ 曝光获得率 <5%：保持象征性出价（仅作流量探测）

        边缘词（低相关性-低流量）
        ┗ 任何情况：直接暂停 + 加入否定词库（除非新品拓词期）
        '''
        advices = [
            '拓展ASIN投放,防御性ASIN否定',
            '保持关键词竞价,动态监测转化率',
            '提升关键词竞价,精准ASIN投放',
            '全面加强关键词投放,精准ASIN投放,检查Listing相关性',
            '优化转化率 > 拓展流量:因流量天花板低',
            '测试长尾词变体投放:包含单复数/错别字',
            '提升长尾词竞价,拓展同义词库',
            '启用ASIN定向收割,短语匹配',
            '采用动态竞价:仅降低,精准匹配',
            '选择性投放:需配合转化率数据判断',
            '选择性埋词优化提升相关性',
            '梯度竞价测试:寻找最佳ROAS点',
            '暂停投放:需先优化产品属性匹配度',
            '立即设置否定关键词',
            '观察转化率:高于平均ACOS则否定',
            '保持象征性出价:仅作流量探测',
            '直接暂停,加入否定词库:除非新品拓词期'
        ]
        advices_hk = [
            '拓展ASIN投放，防禦性ASIN否定',
            '保持關鍵詞競價，動態監測轉化率',
            '提升關鍵詞競價，精準ASIN投放',
            '全面加強關鍵詞投放，精準ASIN投放，檢查Listing相關性',
            '優化轉化率 > 拓展流量：因流量天花板低',
            '測試長尾詞變體投放：包含單複數/錯別字',
            '提升長尾詞競價，拓展同義詞庫',
            '啟用ASIN定向收割，短語匹配',
            '採用動態競價：僅降低，精準匹配',
            '選擇性投放：需配合轉化率數據判斷',
            '選擇性埋詞優化提升相關性',
            '梯度競價測試：尋找最佳ROAS點',
            '暫停投放：需先優化產品屬性匹配度',
            '立即設置否定關鍵詞',
            '觀察轉化率：高於平均ACOS則否定',
            '保持象徵性出價：僅作流量探測',
            '直接暫停，加入否定詞庫：除非新品拓詞期'
        ]
        try:
            data['review_ratio'] = data['calculated_weekly_searches'] * 4 / data['searches']
            conditions = [
               ((data['KeywordClassifier_name'] == '核心热词') & (data['review_ratio'] > 0.3)),
               ((data['KeywordClassifier_name'] == '核心热词') & (data['review_ratio'] <= 0.3) & (
                       data['review_ratio'] > 0.15)),
               ((data['KeywordClassifier_name'] == '核心热词') & (data['review_ratio'] <= 0.15) & (
                       data['review_ratio'] > 0.05)),
               ((data['KeywordClassifier_name'] == '核心热词') & (data['review_ratio'] <= 0.05)),
               ((data['KeywordClassifier_name'] == '精准长尾') & (data['review_ratio'] > 0.6)),
               ((data['KeywordClassifier_name'] == '精准长尾') & (data['review_ratio'] <= 0.6) & (
                           data['review_ratio'] > 0.3)),
               ((data['KeywordClassifier_name'] == '精准长尾') & (data['review_ratio'] <= 0.3)),
               ((data['KeywordClassifier_name'] == '拓展热词') & (data['review_ratio'] > 0.2)),
               ((data['KeywordClassifier_name'] == '拓展热词') & (data['review_ratio'] <= 0.2) & (
                           data['review_ratio'] > 0.1)),
               ((data['KeywordClassifier_name'] == '拓展热词') & (data['review_ratio'] <= 0.1)),
               ((data['KeywordClassifier_name'] == '潜力词') & (data['review_ratio'] > 0.5)),
               ((data['KeywordClassifier_name'] == '潜力词') & (data['review_ratio'] <= 0.5) & (
                           data['review_ratio'] > 0.2)),
               ((data['KeywordClassifier_name'] == '潜力词') & (data['review_ratio'] <= 0.2)),
               ((data['KeywordClassifier_name'] == '泛流量词') & (data['review_ratio'] > 0.1)),
               ((data['KeywordClassifier_name'] == '泛流量词') & (data['review_ratio'] <= 0.1) & (
                       data['review_ratio'] > 0.05)),
               ((data['KeywordClassifier_name'] == '泛流量词') & (data['review_ratio'] <= 0.05)),
               ((data['KeywordClassifier_name'] == '边缘词'))
            ]
            data['Keyword_advice'] = np.select(conditions, advices, default='未分类')
            data['Keyword_advice_hk'] = np.select(conditions, advices_hk, default='未分类')

            return data, None
        except Exception as e:
            logging.error(f"建议分类失败:{str(e)}")
            message = f"建议分类失败:{str(e)}"
            return None, message

    def save_to_database(self, asin):
        """
        :return: 0 -> 某一步失败
        -1 -> 没有参考的卖家精灵数据
        -2 -> 卖家精灵收录的关键词太少
        1 -> 保存成功
        """
        try:
            # TODO 1 首先应该是先获取商品基本信息
            logging.info(f"获取商品基本信息")
            asin_info_s = time.time()
            data, message = self.get_asin_info(asin)
            if not data:
                return 0, message
            asin_info_e = time.time()
            logging.info(f"get_asin_info cost: {asin_info_e - asin_info_s}s")
            # TODO 2 通过大模型分析详情页信息得到初步核心关键词
            logging.info(f"大模型生成核心关键词")
            main_word_s = time.time()
            main_keyword, message = self.get_main_word(data[0]['title'])
            if not main_keyword:
                return 0, message
            logging.info(f"LLM generate: {main_keyword}")
            main_word_e = time.time()
            logging.info(f"LLM generate cost:{main_word_e - main_word_s}s")

            # TODO 3 获取卖家精灵数据
            logging.info(f"获取卖家精灵数据")
            seller_data_s = time.time()
            seller_data_json, message = self.fetch_seller_data(asin)
            if seller_data_json['data']['items'] is None:
                logging.warning(f"{asin}无卖家精灵关键词数据，直接更新核心关键词")
                success, message = self.save_llm_mainword(asin, main_keyword)
                return success, message
            elif len(seller_data_json['data']['items'])<=5:
                logging.warning(f"{asin}无卖家精灵关键词数据，直接更新核心关键词")
                success, message = self.save_llm_mainword(asin, main_keyword)
                return success, message
            elif seller_data_json == 0:
                logging.error(f"{asin}获取卖家精灵数据失败")
                message = f"{asin}获取卖家精灵数据失败"
                return 0, message
            seller_data = seller_data_json.get('data', {})
            if seller_data is None:
                logging.error(f"卖家精灵相关数据解析失败,无法处理ASIN{asin}")
                message = f"卖家精灵相关数据解析失败,无法处理ASIN{asin}"
                return 0, message
            seller_data_e = time.time()
            logging.info(f"seller_data cost:{seller_data_e - seller_data_s}s")

            # TODO 4 获取卖家精灵数据
            logging.info(f"更新卖家精灵相关数据到数据库")
            update_seller_s = time.time()
            seller_success, message = self.batch_update_seller_data(seller_data, asin)
            if seller_data.get('total', 0) < 6:
                return -2, message
            if not seller_success:
                return seller_success, message
            update_seller_e = time.time()
            logging.info(f"update_seller_data cost:{update_seller_e - update_seller_s}s")
            caculate_data, search_result_id = message


            # TODO 5 计算相关性，流量性
            # 数据整合
            calculate_data_s = time.time()
            data, message = self.get_calculate_data(data, caculate_data, search_result_id, asin)
            if data is None or (hasattr(data, 'empty') and data.empty):
                return 0, message
            calculate_data_e = time.time()
            logging.info(f"calculate_data cost:{calculate_data_e - calculate_data_s}s")

            # 开始计算
            keywords = data['keyword'].tolist()
            keyword_id = data['id'].tolist()

            relevance_s = time.time()
            relevance_scores, message = self.calculate_relevance_scores(asin, data['title'][0],
                                                                        data['about_this_item'][0],
                                                                        keywords,
                                                                        main_keyword,
                                                                        )
            if relevance_scores is None:
                return 0, message
            data['correlation'] = relevance_scores
            relevance_scores_avg = np.mean(relevance_scores)
            relevance_e = time.time()
            logging.info(f"calculate_relevance cost:{relevance_e - relevance_s}s")

            classifier_s = time.time()
            data, message = self.classification(data, relevance_scores)
            if data is None or (hasattr(data, 'empty') and data.empty):
                return 0, message
            classifier_e = time.time()
            logging.info(f"calculate_classification: {classifier_e - classifier_s}s")
            advice_s = time.time()
            data, message = self.keyword_advice_classification(data)
            if data is None:
                return 0, message
            advice_e = time.time()
            logging.info(f"advice cost:{advice_e - advice_s}s")
            # 更新数据
            save_s = time.time()
            success, message = self.batch_update_data(data, relevance_scores_avg, min(keyword_id), asin)
            if not success:
                return success, message
            save_e = time.time()
            logging.info(f"save_data: {save_e - save_s}s")
            logging.info(f'api保存 {asin} 信息成功')
            return data, 1
        except Exception as e:
            logging.error(f"主循环错误:{str(e)}")
            message = f"主循环错误:{str(e)}"
            return 0, message

    def calculate_call_back_data(self, data):
        """
        data:{'id, title, des, correlation, search, cal_search, 相关性标签， 流量性标签，关键词分类名称'}
        """
        try:
            classfication = ['核心热词', '精准长尾', '拓展热词', '潜力词', '泛流量词', '边缘词']
            # data = data.fillna({
            #     'searches': 0,
            #     'correlation': 0,
            # })
            class_stats = {}

            total_count = len(data)

            # total_impressions = data['searches'].sum() if 'searches' in data.columns else 0
            total_impressions = data['searches'].fillna(0).sum() if 'searches' in data.columns else 0
            for _, row in data.iterrows():
                class_name = row.get('KeywordClassifier_name', '其他') if hasattr(row, 'get') else row[
                    'KeywordClassifier_name'] if 'KeywordClassifier_name' in row else '其他'
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        "keywordClassName": class_name,
                        "count": 0,
                        "impressions": 0,
                        "correlation": 0,  # Will track sum for now, calculate average later
                        "count_percent": 0,
                        "impressions_percent": 0,
                        "_correlation_sum": 0  # Helper field for calculating average
                    }
                # Update statistics
                class_stats[class_name]["count"] += 1
                class_stats[class_name]["impressions"] += row.get('calculated_weekly_searches', 0) or 0  # 处理None和NaN
                class_stats[class_name]["_correlation_sum"] += row.get('correlation', 0) or 0  # 处理None和NaN
                # class_stats[class_name]["impressions"] += row['searches'] if 'searches' in row else 0
                # class_stats[class_name]["_correlation_sum"] += row['correlation'] if 'correlation' in row else 0

            results = []
            for class_name, stats in class_stats.items():
                stats['impressions'] = int(stats['impressions'])
                if stats["count"] > 0:
                    stats["correlation"] = round(stats["_correlation_sum"] / stats["count"], 2)
                # Calculate percentages
                stats["count_percent"] = round((stats["count"] / total_count) * 100, 2) if total_count > 0 else 0
                stats["impressions_percent"] = round((stats["impressions"] / total_impressions) * 100,
                                                     2) if total_impressions > 0 else 0

                # Remove the helper field
                del stats["_correlation_sum"]

                # Add to results
                results.append(stats)
            return results, None
        except Exception as e:
            logging.error(f"生成回传数据失败:{str(e)}")
            message = f"生成回传数据失败:{str(e)}"
            return None, message

    def run(self, asin):
        """
        数据返回有三种结果可能
        1. 某一个阶段（获取数据 or 计算失败）
        data = 0, message = 错误信息
        2. 没有卖家精灵相关数据
        data = -1, message = 是否成功直接更新大模型提取的核心关键词
        3. 卖家精灵中的关键词收录太少了，存库后不进行计算
        data = -2, message = 卖家精灵api保存
        4. 有卖家精灵相关数据，进行计算成功
        data = 计算，分类成功数据, message = 1
        """
        with redis.Redis(connection_pool=self.pool) as r:
            cached_data = r.get(asin)
        cached_data = json.loads(cached_data) if cached_data else None
        if cached_data:
            logging.info(f"Stream缓存命中 | ASIN: {asin}")
            return cached_data, -1
        data, message = self.save_to_database(asin)
        if isinstance(data, int):
            if data == 0:
                return {
                           "code": 400,
                           "message": message,
                           "data": []
                       }, 0
            elif data == -1:
                return {
                           "code": 500,
                           "message": f"{asin}没有卖家精灵相关数据, 无法分析",
                           "data": []
                       }, -1
            elif data == -2:
                return {
                           "code": 504,
                           "message": f"{asin}关键词收录数量过少，建议参考可作为短期销售目标的竞品ASIN",
                           "data": []
                       }, -2
        else:
            call_back_data, message = self.calculate_call_back_data(data)
            if call_back_data is None:
                return {
                    "code": 400,
                    "message": message,
                    "data": []
                }, 0
            else:
                return {
                    "code": 200,
                    "message": "Success",
                    "data": call_back_data
                }, 1


class AmazonKeywordRanker:
    def __init__(self, db_config: dict, site: str):
        """
        初始化关键词排序处理器
        :param db_config: 数据库配置字典，需包含host, user, password, database
        """
        self.db_config = db_config
        self.site = site
        # 强制设置字符集防止编码问题
        self.db_config.setdefault('charset', 'utf8mb4')
        # 设置连接超时时间为10秒
        self.db_config.setdefault('connect_timeout', 10)

    def _get_db_connection(self):
        """创建新的数据库连接"""
        return pymysql.connect(**self.db_config)

    def update_core_keywords(self, target_asin: str) -> None:
        """
        核心入口方法：更新指定ASIN的核心关键词
        完整流程：获取最新ASIN记录 → 查询关联关键词 → 三级排序 → 更新数据库
        """
        try:
            # 获取最新ASIN记录的ID
            asin_id = self._fetch_latest_asin_id(target_asin)

            # 获取关联关键词数据集
            keywords = self._fetch_asin_keywords(asin_id, target_asin)

            # 执行三级优先级排序
            top_keywords = self._perform_ranking(keywords)

            # 更新核心关键词到数据库
            self._update_core_keywords(target_asin, top_keywords)

        except pymysql.MySQLError as e:
            print(f"[数据库错误] 代码:{e.args[0]} 信息:{e.args[1]}")
            raise
        except ValueError as e:
            print(f"[业务逻辑错误] {str(e)}")
            raise
        except Exception as e:
            print(f"[系统异常] {str(e)}")
            raise

    def hash_mod3(self, asin):
        """对asin哈希取模分表"""
        hash_hex = hashlib.md5(asin.encode('utf-8')).hexdigest()
        # 转为十进制整数
        hash_int = int(hash_hex, 16)
        # 对 3 取模
        return hash_int % 5

    def _fetch_latest_asin_id(self, asin: str) -> int:
        """获取指定ASIN的最新记录ID"""
        conn = self._get_db_connection()
        try:
            table_num = self.hash_mod3(asin)
            with conn.cursor() as cursor:
                query = f"""
                    SELECT id 
                    FROM uatu_keyword_search_results_{self.site}_{table_num}
                    WHERE asin = %s
                    ORDER BY created_at DESC 
                    LIMIT 1
                """
                cursor.execute(query, (asin,))
                result = cursor.fetchone()

                if not result:
                    raise ValueError(f"ASIN {asin} 不存在或没有相关记录")
                return result[0]
        finally:
            conn.close()

    def _fetch_asin_keywords(self, search_result_id: int, asin: str) -> List[Dict]:
        """获取关联关键词数据集"""
        conn = self._get_db_connection()
        try:
            table_num = self.hash_mod3(asin)
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                query = f"""
                    SELECT 
                        keyword,
                        correlation,
                        KeywordClassifier_tr AS traffic_label,
                        KeywordClassifier_re AS relevance_binary
                    FROM uatu_keyword_items_{self.site}_{table_num}
                    WHERE search_result_id = %s
                """
                cursor.execute(query, (search_result_id,))
                results = cursor.fetchall()

                if not results:
                    raise ValueError("未找到关联关键词数据")
                return results
        finally:
            conn.close()

    def _perform_ranking(self, keywords: List[Dict]) -> List[Optional[str]]:
        """执行三级优先级排序算法"""
        # 数据完整性校验
        self._validate_keyword_data(keywords)

        # 三级排序：相关性二分类(降) → 流量标签(降) → 相关性得分(降)
        sorted_keywords = sorted(
            keywords,
            key=lambda x: (
                -x['relevance_binary'],
                -x['traffic_label'],
                -x['correlation']
            )
        )

        # 提取前3名关键词，自动处理不足情况
        return self._format_top_keywords(sorted_keywords)

    def _validate_keyword_data(self, keywords: List[Dict]) -> None:
        """验证关键词数据有效性"""
        required_fields = {
            'keyword',
            'correlation',
            'traffic_label',
            'relevance_binary'
        }

        for kw in keywords:
            missing = required_fields - kw.keys()
            if missing:
                raise KeyError(f"关键词数据缺少必要字段: {missing}")

            # 验证字段取值范围
            if not kw['correlation'] <= 1:
                raise ValueError(f"无效的相关性得分: {kw['correlation']}")
            if kw['traffic_label'] not in {0, 1}:
                raise ValueError(f"无效的流量标签: {kw['traffic_label']}")
            if kw['relevance_binary'] not in {-1, 0, 1}:
                raise ValueError(f"无效的相关性二分类值: {kw['relevance_binary']}")

    def _format_top_keywords(self, sorted_list: List[Dict]) -> List[Optional[str]]:
        """格式化前3名关键词（自动补位）"""
        top_keywords = [item['keyword'] for item in sorted_list[:3]]
        # 保证返回3个元素，不足时用None填充
        return top_keywords + [None] * (3 - len(top_keywords))

    def _update_core_keywords(self, asin: str, keywords: List[str]) -> None:
        """安全更新核心关键词（带数据变化检测）"""
        conn = self._get_db_connection()
        try:
            table_num = self.hash_mod3(asin)
            with conn.cursor() as cursor:
                # 阶段1：验证ASIN存在性
                cursor.execute(f"SELECT 1 FROM uatu_asin_info_{self.site}_{table_num} WHERE asin = %s", (asin,))
                if not cursor.fetchone():
                    print(f"错误终止: ASIN {asin} 不存在")
                    return

                # 阶段2：获取当前关键词状态
                cursor.execute(
                    f"SELECT core_keyword_1, core_keyword_2, core_keyword_3 FROM uatu_asin_info_{self.site}_{table_num} WHERE asin = %s",
                    (asin,)
                )
                current_keys = [kw if kw is not None else '' for kw in cursor.fetchone()]  # 处理NULL值

                # 阶段3：准备新关键词（规范化None为空字符串）
                new_keys = [kw if kw is not None else '' for kw in keywords[:3]]

                # 判断是否需要更新
                if current_keys == new_keys:
                    print(f"提示: ASIN {asin} 的核心关键词无需更新")
                    return

                # 阶段4：执行更新
                update_sql = f"""
                    UPDATE uatu_asin_info_{self.site}_{table_num}
                    SET core_keyword_1 = %s, core_keyword_2 = %s, core_keyword_3 = %s 
                    WHERE asin = %s
                """
                cursor.execute(update_sql, new_keys + [asin])
                conn.commit()
                print(f"成功更新ASIN {asin} 的核心关键词")

        except pymysql.Error as e:
            conn.rollback()
            raise RuntimeError(f"数据库错误: {e.args[1]}")
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()



