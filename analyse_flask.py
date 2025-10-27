from flask import Flask, request, jsonify
import time
import pymysql
import re
from keyword_classifier import KeyWord, AmazonKeywordRanker
from competitors import KeywordsFocusProducts, Config
import redis
import logging
from logging.handlers import TimedRotatingFileHandler
import os

# --- 日志配置 ---
log_file = "/amazon_helper.log"
log_dir = os.path.dirname(log_file)
os.makedirs(log_dir, exist_ok=True)

handler = TimedRotatingFileHandler(
    log_file,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.info("AnalyseFlask启动！")

# --- Flask app ---
app = Flask(__name__)

# --- 多站点配置 ---
SITE_CONFIGS = {
    'us': {
        'DB_HOST': '',
        'DB_USER': '',
        'DB_PASSWORD': '',
        'DB_NAME': '',
        'MODEL_PATH': '/all-MiniLM-L6-v2',
        'REDIS_HOST': '',
        'REDIS_PORT': 6379,
        'REDIS_PASSWORD': '',
        'REDIS_DB': 10,
    },
    'ca': {
        'DB_HOST': '',
        'DB_USER': '',
        'DB_PASSWORD': '',
        'DB_NAME': '',
        'MODEL_PATH': '/all-MiniLM-L6-v2',
        'REDIS_HOST': '',
        'REDIS_PORT': 6379,
        'REDIS_PASSWORD': '',
        'REDIS_DB': 11,
    },
    'uk': {
            'DB_HOST': '',
            'DB_USER': '',
            'DB_PASSWORD': '',
            'DB_NAME': '',
            'MODEL_PATH': '/all-MiniLM-L6-v2',
            'REDIS_HOST': '',
            'REDIS_PORT': 6379,
            'REDIS_PASSWORD': '',
            'REDIS_DB': 12,
        },
    'fr': {
            'DB_HOST': '',
            'DB_USER': '',
            'DB_PASSWORD': '',
            'DB_NAME': '',
            'MODEL_PATH': '/all-MiniLM-L6-v2',
            'REDIS_HOST': '',
            'REDIS_PORT': 6379,
            'REDIS_PASSWORD': '',
            'REDIS_DB': 13,
        },
    'de': {
            'DB_HOST': '',
            'DB_USER': '',
            'DB_PASSWORD': '',
            'DB_NAME': '',
            'MODEL_PATH': '/all-MiniLM-L6-v2',
            'REDIS_HOST': '',
            'REDIS_PORT': 6379,
            'REDIS_PASSWORD': '',
            'REDIS_DB': 13,
        }
}

# 缓存每个站点的组件实例
component_cache = {}

def init_components(site_code: str):
    if site_code not in SITE_CONFIGS:
        raise ValueError(f"站点 '{site_code}' 配置未定义")

    # 已初始化，直接返回缓存组件
    if site_code in component_cache:
        return component_cache[site_code]

    config = SITE_CONFIGS[site_code]

    db_config = {

        'host': config['DB_HOST'],
        'user': config['DB_USER'],
        'password': config['DB_PASSWORD'],
        'database': config['DB_NAME'],
    }

    redis_pool = redis.ConnectionPool(
        host=config['REDIS_HOST'],
        port=config['REDIS_PORT'],
        password=config['REDIS_PASSWORD'],
        db=config['REDIS_DB'],
        decode_responses=False,
        socket_keepalive=True,
        health_check_interval=300,
        encoding='utf-8'
    )

    keyword_processor = KeyWord(db_config,  site_code,config['MODEL_PATH'], redis_pool)
    rank_processor = AmazonKeywordRanker(db_config, site_code)
    competitor_processor = KeywordsFocusProducts(Config(
        host=config['DB_HOST'],
        user=config['DB_USER'],
        password=config['DB_PASSWORD']
    ), site_code)

    components = {
        'keyword_processor': keyword_processor,
        'rank_processor': rank_processor,
        'competitor_processor': competitor_processor,
    }
    component_cache[site_code] = components
    return components

# --- 路由 ---

@app.route('/process/<string:asin>/<string:site_code>', methods=['GET'])
def process(site_code, asin):
    try:
        logger.info(f"收到处理请求 | ASIN: {asin}, Site: {site_code}")
        if site_code not in SITE_CONFIGS:
            return jsonify({"code": "400", "message": f"未知站点: {site_code}"}), 400

        if not re.match(r'^[A-Z0-9]{10}$', asin):
            return jsonify({
                "code": "400",
                "message": "无效的ASIN格式，必须为10位大写字母数字组合"
            }), 400

        components = init_components(site_code)
        keyword_processor = components['keyword_processor']
        rank_processor = components['rank_processor']
        logger.info("开始调用 keyword_processor.run")
        json_data, flag = keyword_processor.run(asin)
        logger.info("keyword_processor.run 完成")
        if flag == 1:
            logger.info("开始 update_core_keywords")
            rank_processor.update_core_keywords(asin)
            logger.info("update_core_keywords 完成")

        return jsonify(json_data)

    except Exception as e:
        logger.error(f"[{site_code}] process接口异常: {e}", exc_info=True)
        return jsonify({
            'code': '500',
            'message': f"{site_code}主循环错误：{str(e)}",
            'asin': asin
        }), 500


@app.route('/competitors/<string:ad_asin>/<string:keyword>/<string:site_code>', methods=['GET'])
def competitors(site_code, ad_asin, keyword):
    start_time = time.time()
    try:
        if site_code not in SITE_CONFIGS:
            return jsonify({"code": 400, "message": f"未知站点: {site_code}", "data": []}), 400

        if not ad_asin or not keyword:
            return jsonify({
                "code": 1001,
                "message": "无效参数格式，需要: ad_asin和keyword",
                "data": []
            }), 400

        if not re.match(r'^[A-Z0-9]{10}$', ad_asin):
            return jsonify({
                "code": 1002,
                "message": "无效的ASIN格式，必须为10位大写字母数字组合",
                "data": []
            }), 400

        components = init_components(site_code)
        competitor_processor = components['competitor_processor']

        success, message, back_data, time_json_data, product_data = competitor_processor.run(ad_asin, keyword)
        processing_time = round(time.time() - start_time, 2)

        return jsonify({
            "code": 200 if success else 500,
            "message": message,
            "data": {
                "competitors_data": back_data,
                "price_changes": time_json_data,
                "price": product_data['price'],
                "rating": product_data['rating'],
                "reviewCount": product_data['review_count']
            },
            "processing_time": processing_time
        }), 200 if success else 500

    except Exception as e:
        logger.error(f"[{site_code}] competitors接口异常: {e}", exc_info=True)
        return jsonify({
            "code": 1004,
            "message": f"{site_code}服务器内部错误: {str(e)}",
            "data": []
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

