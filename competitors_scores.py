import logging

from flask import Flask, request, jsonify
import pymysql
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
import torch
from scipy.spatial.distance import cosine
from datetime import datetime
from typing import Dict, Optional, Tuple, List, TypedDict
from contextlib import contextmanager
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache
from urllib.parse import unquote
import json


# ======================
# 类型定义
# ======================
class AsinInfo(TypedDict):
    asin: str
    title: str
    core_keywords: List[str]
    price: float
    rating: float
    review_count: int
    created_at: datetime


class RankFeatures(TypedDict):
    final_rank_score: float
    latest_rank: int
    decay_weight: float


# ======================
# 配置类
# ======================
class Config:
    # DB_CONFIG = {
    #     'host': '192.168.1.4',
    #     'user': 'mxk',
    #     'password': 'Mm712819830.',
    #     'database': 'amazon_asins',
    #     'charset': 'utf8mb4',
    #     'cursorclass': pymysql.cursors.DictCursor
    # }
    DB_CONFIG = {
        'host': 'rm-bp1n5rqs1z21tbz965o.mysql.rds.aliyuncs.com',
        'user': 'Uatu',
        'password': 'UatuAI@Aliyun2025',
        'database': 'amazon_asins_ca',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    USE_GPU = True
    MAX_URL_LENGTH = 2000
    REQUEST_TIMEOUT = 30


# ======================
# 核心计算类
# ======================
class FeatureWeights:
    def __init__(self):
        self.similarity_d = 0.5
        self.price = 0.15
        self.rating = 0.10
        self.review = 0.05
        self.rank = 0.2
        self.position = 0.1

        self.sponsor_mix = 0.65
        self.nature_mix = 0.35
        self.sponsor_solo = 1.3
        self.nature_solo = 1.1


class AsinSimilarityCalculator:
    def __init__(self, db_config: Dict, model_name: str = Config.MODEL_NAME, use_gpu: bool = Config.USE_GPU):
        self.db_config = db_config
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, self.device)
        self.model.eval()
        self.feature_weights = FeatureWeights()
        self.embed_cache = LRUCache(maxsize=10000)

    def clamp(self, data):
        data = np.where(data < 0.0, 0.0, data)
        data = np.where(data > 1.0, 1.0, data)
        return data

    @contextmanager
    def _db_connection(self):
        conn = pymysql.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def _execute_query(self, query: str, params: tuple) -> Optional[dict]:
        """统一执行查询"""
        try:
            with self._db_connection() as conn:
                with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchone()
        except Exception as e:
            print(f"[DB ERROR] Query failed: {str(e)}")
            return None

    # def get_main_asin_info(self, main_asin: str) -> Optional[AsinInfo]:
    #     """获取本品信息（带缓存）"""
    #     query = """
    #         SELECT asin, title,
    #             core_keyword_1, core_keyword_2, core_keyword_3,
    #             price, rating, review_count, created_at
    #         FROM amazon_asins.uatu_asin_info
    #         WHERE asin = %s
    #         ORDER BY created_at DESC
    #         LIMIT 1;
    #     """
    #     result = self._execute_query(query, (main_asin,))
    #     if not result:
    #         print(f"[ERROR] 主品ASIN {main_asin} 未找到")
    #         return None
    #
    #     core_keywords = [
    #         result.get('core_keyword_1', '').strip(),
    #         result.get('core_keyword_2', '').strip(),
    #         result.get('core_keyword_3', '').strip()
    #     ]
    #     result['core_keywords'] = [kw for kw in core_keywords if kw]
    #     return result

    def calculate_average_rank(self, data):
        """计算平均排名"""
        if not data:
            return
        valid_ranks = [item['rank'] for item in data if item['rank'] is not None]
        if not valid_ranks:
            return None

        return math.ceil(sum(valid_ranks) / len(valid_ranks))

    def get_comp_info(self, comp_infos):
        title = []
        price = []
        rating = []
        review_count = []
        ad_rank = []
        natural_rank = []
        for asin, data_group in comp_infos.items():
            title.append(data_group['latest_data']['title'])
            price.append(
                float(data_group['latest_data']['price']) if data_group['latest_data']['price'] is not None else 0.0)
            rating.append(
                float(data_group['latest_data']['rating']) if data_group['latest_data']['rating'] is not None else 0.0)
            review_count.append(int(data_group['latest_data']['review_count']) if data_group['latest_data'][
                                                                                      'review_count'] is not None else 0)
            ad_rank.append(self.calculate_average_rank(data_group['sponsored_data']))
            natural_rank.append(self.calculate_average_rank(data_group['natural_data']))

        return title, price, rating, review_count, ad_rank, natural_rank

    def calculate_core_similarity(self, main_title_vec, comp_title_vecs, core_keywords):
        if core_keywords:
            weights = torch.tensor([0.5, 0.3, 0.3]).to(self.device).unsqueeze(1)
            core_vec = self.model.encode(core_keywords, convert_to_tensor=True)
            core_vec = torch.sum(core_vec * weights, dim=0)

            sim_main = util.pytorch_cos_sim(core_vec, main_title_vec).squeeze(0).squeeze(0)
            sim_comp = util.pytorch_cos_sim(core_vec.unsqueeze(0), comp_title_vecs).squeeze(0)

            raw_score = (sim_comp / (sim_main + 1e-6)) ** 0.8
            return raw_score / (1 + raw_score)

        return util.pytorch_cos_sim(main_title_vec, comp_title_vecs.unsqueeze(0))

    def calculate_abc_similarity(self, main_info, comp_titles, keyword):
        """ABC三类相似度计算"""
        # titles = [main_info['title'], comp_info['title'], keyword]
        # main_vec, comp_vecs, kw_vec = self.batch_embed_texts()
        main_title_vec = self.model.encode(main_info['title'], convert_to_tensor=True)
        comp_title_vecs = self.model.encode(comp_titles, convert_to_tensor=True)

        kw_vec = self.model.encode(keyword, convert_to_tensor=True)

        similarity_a = self.calculate_core_similarity(
            main_title_vec, comp_title_vecs,
            main_info.get('core_keywords', [])
        ).cpu().detach().numpy()
        keyword_main_similarity = util.pytorch_cos_sim(main_title_vec, kw_vec).squeeze(0).squeeze(
            0).cpu().detach().numpy()

        keyword_comp_similarity = util.pytorch_cos_sim(kw_vec.unsqueeze(0), comp_title_vecs).squeeze(
            0).cpu().detach().numpy()

        ratio = (keyword_comp_similarity + 1e-6) / (keyword_main_similarity + 1e-6)
        similarity_b = self.clamp((2 / (1 + np.exp(-3 * (ratio - 1)))) / 2)

        similarity_c = util.pytorch_cos_sim(main_title_vec.unsqueeze(0),
                                            comp_title_vecs).squeeze().cpu().detach().numpy()

        similarity_d = (similarity_a * 0.4) + (similarity_b * 0.3) + (similarity_c * 0.3)
        logging.info("similarity_d_pass")

        return similarity_a, similarity_b, similarity_c, similarity_d

    def calculate_price_similarity(self, price1, price2, max_price_id, min_price_id, k=4):
        """
        计算价格相似度
        :param price1: 本品价格
        :param price2: 竞品价格数组
        :param max_price: 用户设置的最高价格
        :param min_price: 用户设置的最低价格
        :param k: k越大价格相似度对价格差异越敏感
        :return: 价格相似度
        """
        price2 = np.array(price2, dtype=float)
        result = np.zeros_like(price2, dtype=float)

        # 处理price1 <= 0的情况
        if price1 <= 0:
            return result

        # 使用numpy的where函数处理price2 <= 0的情况
        valid_mask = price2 >= 0

        # 只对有效的元素进行计算
        valid_price2 = price2[valid_mask]
        min_price = np.min(valid_price2)
        max_price = np.max(valid_price2)
        normal_price2 = 10 * (price2 - min_price) / (max_price - min_price)

        # 对price1进行同样的归一化处理
        normal_price1 = 10 * (price1 - min_price) / (max_price - min_price)
        if len(valid_price2) > 0:
            differences = np.abs(normal_price1 - normal_price2) / (normal_price1 + normal_price2)
            valid_result = np.exp(-k * differences)

            # 应用clamp到有效结果
            valid_result = self.clamp(valid_result)

            # 将结果放回原数组
            result[valid_mask] = valid_result

            if max_price_id is not None and min_price_id is not None and max_price_id > min_price_id:
                max_price_id = float(max_price_id)
                min_price_id = float(min_price_id)
                price_interval = 0.25 * (max_price_id - min_price_id)
                price_mask = np.zeros(len(result))
                index_inside = np.where((price2 >= (min_price_id - price_interval)) & (price2 <= (max_price_id + price_interval)))[0]
                price_mask[index_inside] = 1
                result = result * price_mask

        return result

    def calculate_rating_similarity(self, rating1, rating2):
        rating2 = np.array(rating2, dtype=float)
        min_rating = min(max(rating1, 0), 5.0)

        # 计算绝对差值
        delta = np.abs(rating1 - rating2)

        # 计算惩罚值
        penalty = (delta / 4) ** 1.5 * (1 - min_rating / 5)

        # 应用clamp并返回结果
        return self.clamp(1 - penalty)

    def calculate_review_similarity(self, c1, c2):
        c2 = np.array(c2, dtype=float)
        result = np.zeros_like(c2, dtype=float)

        # 处理特殊情况：c1 + c2 = 0
        zeros_mask = (c1 + c2) == 0
        result[zeros_mask] = 1.0

        # 对非零情况进行计算
        non_zeros_mask = ~zeros_mask
        if np.any(non_zeros_mask):
            c2_valid = c2[non_zeros_mask]

            min_c = np.minimum(c1, c2_valid)
            max_c = np.maximum(c1, c2_valid)

            term1 = np.log1p(min_c) / np.log1p(max_c)
            term2 = 1 - np.abs(c1 - c2_valid) / (c1 + c2_valid + 1)

            valid_result = term1 * term2
            result[non_zeros_mask] = valid_result

        # 应用clamp
        return self.clamp(result)

    def calculate_position_similarity(self, own_nature_position, own_ad_position, comp_nature_position, comp_ad_position):
        """
        计算自然位排名和广告位排名相似度
        :param own_nature_position: 本品自然位排名
        :param own_ad_position: 本品广告位排名
        :param comp_nature_position: 竞品自然位排名数组
        :param comp_ad_position: 竞品广告位排名数组
        :return: 本品和竞品的排名相似度数组
        """
        weights = self.feature_weights
        if own_nature_position is not None and own_ad_position is not None:
            weight_own_position = (weights.sponsor_mix * own_ad_position +
                                    weights.nature_mix * own_nature_position)
        elif own_nature_position is not None and own_ad_position is None:
            weight_own_position = weights.nature_solo * own_nature_position
        elif own_nature_position is None and own_ad_position is not None:
            weight_own_position = weights.sponsor_solo * own_ad_position
        else:
            return None

        n = max(len(comp_nature_position), len(comp_ad_position))
        natural_ranks = np.asarray(comp_nature_position, dtype=float)
        sponsored_ranks = np.asarray(comp_ad_position, dtype=float)
        # 扩展数组长度，确保两个数组长度相同
        if len(natural_ranks) < n:
            natural_ranks = np.pad(natural_ranks, (0, n - len(natural_ranks)),
                                   constant_values=np.nan)
        if len(sponsored_ranks) < n:
            sponsored_ranks = np.pad(sponsored_ranks, (0, n - len(sponsored_ranks)),
                                     constant_values=np.nan)

        # 创建结果数组
        weight_ranks = np.full(n, 999)


        # 掩码：两者都有有效值
        both_valid = ~np.isnan(natural_ranks) & ~np.isnan(sponsored_ranks)
        # 掩码：只有自然排名有效
        only_natural = ~np.isnan(natural_ranks) & np.isnan(sponsored_ranks)
        # 掩码：只有广告排名有效
        only_sponsored = np.isnan(natural_ranks) & ~np.isnan(sponsored_ranks)

        # 计算加权排名
        weight_ranks[both_valid] = (weights.sponsor_mix * sponsored_ranks[both_valid] +
                                    weights.nature_mix * natural_ranks[both_valid])
        weight_ranks[only_natural] = weights.nature_solo * natural_ranks[only_natural]
        weight_ranks[only_sponsored] = weights.sponsor_solo * sponsored_ranks[only_sponsored]

        position_sim = np.abs(weight_ranks-weight_own_position)
        position_sim = MinMaxScaler().fit_transform(position_sim.reshape(-1, 1)).flatten()

        return position_sim

    def _filter_rank_outliers(self, ranks: List[int]) -> List[int]:
        """过滤异常波动"""
        if len(ranks) < 2:
            return ranks
        filtered = [ranks[0]]
        for r in ranks[1:]:
            if abs(r - filtered[-1]) <= 50:
                filtered.append(r)
        return filtered

    def calculate_rank_features(self, comp_infos):
        results = []
        for asin, data_group in comp_infos.items():
            if not data_group['all_data']:
                results.append(None)
                continue
            # 提取排名数据
            ranks = [entry['rank'] for entry in data_group['all_data']]
            valid_ranks = self._filter_rank_outliers(ranks)
            # 计算标准差
            std_dev = np.std(valid_ranks) if len(valid_ranks) >= 2 else 0.0
            # 获取最新排名和时间差
            latest_rank = data_group['all_data'][0]['rank']
            time_diff = (datetime.now() - data_group['all_data'][0]['insert_time']).days

            # 计算衰减因子
            lambda_val = 0.1 + 0.3 * math.log(1 + min(std_dev / 20, 1))
            decay_weight = math.exp(-lambda_val * max(time_diff, 0))

            # 计算原始分数
            raw_score = math.exp(-0.15 * (latest_rank - 1)) if latest_rank <= 10 else 1 / (latest_rank ** 0.3)

            # 添加到结果列表
            results.append(RankFeatures(
                final_rank_score=raw_score * decay_weight,
                latest_rank=latest_rank,
                decay_weight=decay_weight
            ))
        return results

    def review_score_function(self, x_array, k=0.1, a=50):
        """
        计算流量评分的函数 - 支持数组输入

        .. math:: f(x)= \\frac{1}{1+e^{k*(x-a)}}

        参数:
            x_array: 加权排名值数组
            k: 曲线的陡峭程度（k越大，曲线下降越陡）
            a: 拐点的位置

        返回:
            numpy.ndarray: 反转的sigmoid值数组 (0-1范围)
        """
        # 确保输入是 numpy 数组
        x_array = np.asarray(x_array, dtype=float)

        # 创建结果数组
        result = np.zeros_like(x_array)

        # 处理非NaN非无穷大的值
        valid_mask = ~(np.isnan(x_array) | np.isinf(x_array))

        # 针对有效值计算sigmoid
        safe_exp = np.clip(k * (x_array[valid_mask] - a), -709, 709)  # 防止exp溢出
        result[valid_mask] = 1 / (1 + np.exp(safe_exp))

        # NaN和无穷大的值保持为0
        return result

    def calculate_review_score(self, natural_ranks, sponsored_ranks):
        """
        计算流量评分 - 支持数组输入

        参数:
            natural_ranks: 竞品自然位排名数组
            sponsored_ranks: 竞品广告位排名数组

        返回:
            numpy.ndarray: 流量评分数组 (0-10范围)

        说明:
            - 如果两个排名的信息都有: 加权排名 = 自然位权重 × 自然位排名 + 广告位权重 × 广告位排名
            - 如果只有自然位排名: 加权排名 = 单独自然位权重 × 自然位排名
            - 如果只有广告位排名: 加权排名 = 单独广告位权重 × 广告位排名
        """
        # 确保输入是numpy数组
        natural_ranks = np.asarray(natural_ranks, dtype=float)
        sponsored_ranks = np.asarray(sponsored_ranks, dtype=float)

        weights = self.feature_weights
        n = max(len(natural_ranks), len(sponsored_ranks))

        # 扩展数组长度，确保两个数组长度相同
        if len(natural_ranks) < n:
            natural_ranks = np.pad(natural_ranks, (0, n - len(natural_ranks)),
                                   constant_values=np.nan)
        if len(sponsored_ranks) < n:
            sponsored_ranks = np.pad(sponsored_ranks, (0, n - len(sponsored_ranks)),
                                     constant_values=np.nan)

        # 创建结果数组
        weight_ranks = np.full(n, 999)

        # 掩码：两者都有有效值
        both_valid = ~np.isnan(natural_ranks) & ~np.isnan(sponsored_ranks)
        # 掩码：只有自然排名有效
        only_natural = ~np.isnan(natural_ranks) & np.isnan(sponsored_ranks)
        # 掩码：只有广告排名有效
        only_sponsored = np.isnan(natural_ranks) & ~np.isnan(sponsored_ranks)

        # 计算加权排名
        weight_ranks[both_valid] = (weights.sponsor_mix * sponsored_ranks[both_valid] +
                                    weights.nature_mix * natural_ranks[both_valid])
        weight_ranks[only_natural] = weights.nature_solo * natural_ranks[only_natural]
        weight_ranks[only_sponsored] = weights.sponsor_solo * sponsored_ranks[only_sponsored]

        # 处理两者都无效的情况 - 保持为0

        # 计算评分并缩放到0-10
        review_scores = self.review_score_function(weight_ranks) * 10

        return review_scores

    def competitor_classification(self, recommendation_score, review_scores, review_thred=6):
        """
        将竞品分类为六种类型 - 支持数组输入

        参数:
            similarity_scores: 相似度评分数组
            recommendation_score: 推荐评分数组
            review_thred: 高流量阈值 (默认: 6)

        返回:
            numpy.ndarray: 竞品分类结果数组

        分类标准:
            推荐分类根据上下四分位分割
            - 高相关性-高流量：核心竞品
            - 高相关性-低流量：精准竞品
            - 中相关性-高流量：拓展竞品
            - 中相关性-低流量：潜力竞品
            - 低相关性-高流量：泛流量竞品
            - 低相关性-低流量：边缘竞品
        """
        # 竞品分类名称列表
        competitorClassification_list = ['核心竞品', '精准竞品', '拓展竞品', '潜力竞品', '泛流量竞品', '边缘竞品']
        competitorClassification_list_hk = ['核心競品', '精準競品', '拓展競品', '潛力競品', '泛流量競品', '邊緣競品']

        # 确保输入是numpy数组
        recommendation_scores = np.asarray(recommendation_score, dtype=float)
        review_scores = np.asarray(review_scores, dtype=float)



        # 处理NaN值
        recommendation_scores = np.nan_to_num(recommendation_scores, nan=0.0)
        review_scores = np.nan_to_num(review_scores, nan=0.0)

        recommendation_thred_high = np.percentile(recommendation_scores, 85)
        recommendation_thred_low = np.percentile(recommendation_scores, 25)

        conditions = [
            # 1. 核心竞品（高相关 + 高流量）
            (recommendation_scores > recommendation_thred_high) & (review_scores > review_thred),

            # 2. 精准竞品（高相关 + 低流量）
            (recommendation_scores > recommendation_thred_high) & (review_scores <= review_thred),

            # 3. 拓展竞品（中相关 + 高流量）
            (recommendation_scores >= recommendation_thred_low) &
            (recommendation_scores <= recommendation_thred_high) &
            (review_scores > review_thred),

            # 4. 潜力竞品（中相关 + 低流量）
            (recommendation_scores >= recommendation_thred_low) &
            (recommendation_scores <= recommendation_thred_high) &
            (review_scores <= review_thred),

            # 5. 范流量竞品（低相关 + 高流量）
            (recommendation_scores < recommendation_thred_low) & (review_scores > review_thred),

            # 6. 边缘竞品（低相关 + 低流量）
            (recommendation_scores < recommendation_thred_low) & (review_scores <= review_thred)
        ]
        result = np.select(condlist=conditions, choicelist=competitorClassification_list, default='其他')
        result_hk = np.select(condlist=conditions, choicelist=competitorClassification_list_hk, default='其他')

        return result, result_hk

    def comptetitor_advice(self, main_info, comp_prices, scores, comp_classification):
        '''
        核心竞品ASIN（高相关+高流量）
        ┣ 推荐分 >80：SP精准投放（固定建议竞价*1.3） + 进行关键词分析 + 设置防御性ASIN否定（如有产品劣势）
        ┣ 推荐分60-80：SP拓展投放（建议竞价*1.1） + 采集其长尾词 + 动态竞价（仅降低）

        精准竞品ASIN（高相关+低流量）
        ┣ 价格高于本品：SP精准投放（动态建议竞价） + TOS/ROS加价 + 补充QA关键词埋入
        ┗ 价格低于本品：SP精准投放（动态建议竞价*0.9）+  固定竞价

        拓展竞品ASIN（中相关+高流量）
        ┣ 价格低于本品：SP拓展投放（建议竞价*0.7） + SB商品集投放（建议竞价*0.8）+ 动态竞价（仅降低）
        ┗ 价格高与本品：SP精准投放（建议竞价*0.8）+ 动态竞价（仅降低）

        潜力竞品ASIN（中相关+低流量）
        ┣ 价格低于本品：否定投放（在所有广告组中均否定）
        ┗ 价格高与本品：SP拓展投放（建议竞价*0.5）+ 动态竞价（仅降低）

        泛流量竞品ASIN（低相关+高流量）
        ┣ 价格低于本品：否定投放
        ┗ 价格高于本品：SP拓展投放（固定建议竞价*0.4） + SB商品集投放（建议竞价*0.4，自动竞价开）

        边缘竞品ASIN（低相关+低流量）
        ┗ 任何情况：加入否定名单 + 不主动进行投放
        :param main_info:
        :param comp_prices:
        :param scores:
        :param comp_classification:
        :return:data_advice 数组建议分类
        '''

        price1 = float(main_info.get('price', 0)) if main_info.get('price') is not None else 0.0
        data = pd.DataFrame({
            '竞品价格': comp_prices,
            '得分': scores,
            '分类': comp_classification
        })
        advices = [
            'SP精准投放:固定建议竞价*1.3,进行关键词分析,设置防御性ASIN否定:如有产品劣势',
            'SP拓展投放:建议竞价*1.1,采集其长尾词,动态竞价:仅降低',
            'SP精准投放:动态建议竞价,TOS/ROS加价,补充QA关键词埋入',
            'SP精准投放:动态建议竞价*0.9,固定竞价',
            'SP拓展投放:建议竞价*0.7,SB商品集投放:建议竞价*0.8,动态竞价:仅降低',
            'SP精准投放:建议竞价*0.8,动态竞价:仅降低',
            '否定投放:在所有广告组中均否定',
            'SP拓展投放:建议竞价*0.5,动态竞价:仅降低',
            '否定投放',
            'SP拓展投放:固定建议竞价*0.4,SB商品集投放:建议竞价*0.4，自动竞价开',
            '加入否定名单,不主动进行投放'
        ]
        conditions = [
            (data['分类'] == '核心竞品') & (data['得分'] > 80),
            (data['分类'] == '核心竞品') & (data['得分'] >= 60) & (data['得分'] < 80),
            (data['分类'] == '精准竞品') & (data['竞品价格'] > price1),
            (data['分类'] == '精准竞品') & (data['竞品价格'] <= price1),
            (data['分类'] == '拓展竞品') & (data['竞品价格'] <= price1),
            (data['分类'] == '拓展竞品') & (data['竞品价格'] > price1),
            (data['分类'] == '潜力竞品') & (data['竞品价格'] <= price1),
            (data['分类'] == '潜力竞品') & (data['竞品价格'] > price1),
            (data['分类'] == '泛流量竞品') & (data['竞品价格'] <= price1),
            (data['分类'] == '泛流量竞品') & (data['竞品价格'] > price1),
            (data['分类'] == '边缘竞品')
        ]
        advices_hk = [
            'SP精準投放:固定建議競價1.3，進行關鍵詞分析，設置防禦性ASIN否定:如有產品劣勢',
            'SP拓展投放:建議競價1.1，採集其長尾詞，動態競價:僅降低',
            'SP精準投放:動態建議競價，TOS/ROS加價，補充QA關鍵詞埋入',
            'SP精準投放:動態建議競價0.9，固定競價',
            'SP拓展投放:建議競價0.7，SB商品集投放:建議競價0.8，動態競價:僅降低',
            'SP精準投放:建議競價0.8，動態競價:僅降低',
            '否定投放:在所有廣告組中均否定',
            'SP拓展投放:建議競價0.5，動態競價:僅降低',
            '否定投放',
            'SP拓展投放:固定建議競價0.4，SB商品集投放:建議競價*0.4，自動競價開',
            '加入否定名單，不主動進行投放'
        ]
        data_advice = np.select(condlist=conditions, choicelist=advices, default='未分类')
        data_advice_hk = np.select(condlist=conditions, choicelist=advices_hk, default='未分类')

        return data_advice, data_advice_hk

    def calculate_recommendation_score(self, similarity_d, main_info, comp_prices, comp_ratings, comp_reviews,
                                       comp_infos, comp_natural_position, comp_ad_position):
        price1 = float(main_info.get('price', 0)) if main_info.get('price') is not None else 0.0
        logging.info("pc1_pass")

        rating1 = float(main_info.get('rating', 0)) if main_info.get('rating') is not None else 0.0
        logging.info("rt1_pass")

        review1 = int(main_info.get('review_count', 0)) if main_info.get('review_count') is not None else 0
        logging.info("review1_pass")

        max_price = float(main_info.get('max_price', 9999999999)) if main_info.get('max_price') is not None else None
        logging.info("max_price_pass")

        min_price = float(main_info.get('min_price', 0)) if main_info.get('min_price') is not None else None
        logging.info("min_price_pass")

        ad_position = main_info.get('adposition')
        logging.info("ad_position_pass")

        nature_position = main_info.get('natureposition')
        logging.info("nature_position_pass")

        price_sim = self.calculate_price_similarity(price1, comp_prices, max_price, min_price)
        logging.info("price_sim_pass")

        rating_sim = self.calculate_rating_similarity(rating1, comp_ratings)
        logging.info("rating_sim_pass")

        review_sim = self.calculate_review_similarity(review1, comp_reviews)
        logging.info("review_sim_pass")

        position_sim = self.calculate_position_similarity(nature_position, ad_position, comp_natural_position, comp_ad_position)
        if position_sim is None:
            logging.info("本品无自然位和广告位排名信息无法计算排名相似度")
        else:
            logging.info("position_sim_pass")

        rank_features = self.calculate_rank_features(comp_infos)
        logging.info("rank_features_pass")
        rank_sim = np.array(
            [rank_feature['final_rank_score'] if rank_feature else 0.0 for rank_feature in rank_features], dtype=float)
        logging.info(rank_sim)
        logging.info("rank_sim_pass")
        weights = self.feature_weights



        # scores = np.array([
        #     weights.similarity_d * similarity_d + weights.price * price_sim + weights.rating * rating_sim + weights.review + review_sim + weights.rank + rank_sim
        # ])
        if position_sim is None:
            scores = weights.similarity_d * similarity_d + weights.price * price_sim + weights.rating * rating_sim + weights.review * review_sim + weights.rank * rank_sim

        else:
            scores = (weights.similarity_d-0.1) * similarity_d + weights.price * price_sim + weights.rating * rating_sim + weights.review * review_sim + weights.rank * rank_sim + weights.position * position_sim

        scores = self.clamp(scores) * 100
        return scores

    def get_similarity_and_score(self, main_info, comp_infos, keyword):
        # main_info = self.get_main_asin_info(main_asin)
        if not main_info:
            return None

        comp_titles, comp_prices, comp_ratings, comp_review, ad_rank, natural_rank = self.get_comp_info(comp_infos)
        logging.info("get_comp_info_pass")
        # comp_infos = self.get_competing_asin_info(main_asin, competing_asins)  # list
        # if not comp_infos:
        #     return None

        a, b, c, d = self.calculate_abc_similarity(main_info, comp_titles, keyword)
        recommendation_score = self.calculate_recommendation_score(d, main_info, comp_prices, comp_ratings, comp_review, comp_infos, natural_rank, ad_rank)
        logging.info("recommendation_score_pass")
        review_score = self.calculate_review_score(natural_rank, ad_rank)
        competitorClassification, competitorClassification_hk = self.competitor_classification(recommendation_score, review_score)
        data_advice, data_advice_hk = self.comptetitor_advice(main_info, comp_prices, recommendation_score, competitorClassification)
        return a, d, recommendation_score, competitorClassification, competitorClassification_hk, ad_rank, natural_rank, data_advice, data_advice_hk
