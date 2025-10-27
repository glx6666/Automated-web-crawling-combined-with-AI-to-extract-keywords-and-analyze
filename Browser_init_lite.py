from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import ddddocr
import logging
import time
import random
import requests
import re
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import platform
import os
import subprocess
from datetime import datetime
from fake_useragent import UserAgent
from amazoncaptcha import AmazonCaptcha
from bs4 import BeautifulSoup
import json
import logging
import sys
import pandas as pd
from ProductDetailScraper import ProductDetailScraper

file_handler = logging.FileHandler('scraper.log', encoding='utf-8')

# 创建流日志（标准输出），并显式指定 utf-8 编码（避免 GBK 编码报错）
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)

# 多站点配置（示例）
SITE_CONFIGS = {
    "us": {
        "country": "美国",
        "lang": "en-US",
        "product_url": "https://www.amazon.com/dp/{asin}/",
        "asin_list": [
    "B0DD44DHMB",
    "B0073E7JFK",
    "B08TT7Q8YW",
    "B0DY2L7V1R",
    "B0D32J3VNS",
    "B0748W31L5",
    "B085DTZQNZ",
    "B0B2K47S1T",
    "B0BXBJNZYB",
    "B0768N9N6M",
    "B08QLFM4GQ",
    "B08GYFDWHF",
    "B004G4THDM",
    "B082BPQH6Z",
    "B0C145Z4YN",
    "B0DCFDTNRG",
    "B0BLSML9G3",
    "B0BLRPTPYL",
    "B084LKFVS9",
    "B0BJ6V38SZ",
    "B0D3XDP15B",
    "B09VKZMBCX",
    "B0D3J7M411",
    "B0DDBKZVRC",
    "B09NMLH6RQ",
    "B0DGX6H2RQ",
    "B08G9XC8R2",
    "B002YTY4FM",
    "B07M7NTSCN",
    "B08FFCV8TW",
    "B0DGX67FPD",
    "B0D64KTLL9",
    "B00262227E",
    "B0DCF8YKR1",
    "B088D6VLTH",
    "B0BRNY9HZB",
    "B0C3M7N76R",
    "B0D4YK93DV",
    "B000FGECAI",
    "B09QH1S7K8",
    "B08G1XHF76",
    "B0BTPZRJDG",
    "B07VS46CXT",
    "B075ZJ37NR",
    "B09N6Y83N8",
    "B09HR1Q6DP",
    "B0CQ54FM9K",
    "B079CVR1R9",
    "B00BN4QVF0",
    "B0BVLJTXGS"
],
        "zip_format": r"^\d{5}$",  # 美国邮编格式（5位数字）
        "zip_list": [
            '90001',  # 密歇根州
            '10001',  # 纽约州
            '20001',  # 华盛顿特区
            '60102',  # 伊利诺伊州
            '55414',  # 威斯康星州
            '80202',  # 科罗拉多州
            '91301',  # 纽约州
            '94105',  # 加利福尼亚州
            '85298',  # 亚利桑那州
            '75231',  # 德克萨斯州
            '24340',  # 西弗吉尼亚州
            '48104',  # 密歇根州
            '55427',  # 威斯康星州
            '11801',  # 伊利诺伊州
            '55414',  # 威斯康星州
            '80123',  # 科罗拉多州
            '55414',  # 威斯康星州
            '75231',  # 德克萨斯州
            '24340',  # 西弗吉尼亚州
            '48105',  # 密歇根州
            '10002',  # 纽约州
            '55414',  # 威斯康星州
            '91301',  # 纽约州
            '80202',  # 科罗拉多州
            '75231',  # 德克萨斯州
            '24340',  # 西弗吉尼亚州
            '48104'  # 密歇根州
        ],
        "locators": {  # 页面元素定位器（按功能分类）
            "zip_input": (By.ID, "GLUXZipUpdateInput"),
            "LOCATORS_US": [
    (By.XPATH, "//span[text()='Apply']/preceding-sibling::input[@type='submit']"),
    (By.XPATH,
     "//span[@id='GLUXZipUpdate']//input[@class='a-button-input' and @type='submit']"),
    (By.XPATH,
     "//input[@class='a-button-input' and @type='submit']"),
    # 保留原始定位器
]
        },
        "success_text": [
            "We will use your selected location to show all products available for the",
            "我们将使用您选择的位置显示所有可用的产品"
        ]
    },
    "ca": {
        "country": "加拿大",
        "lang": "en-CA",
        "product_url": "https://www.amazon.ca/dp/{asin}/",
        "asin_list": [
    "B09W3Z3QRR",
    "B09SYC5NW2",
    "B0D2M2N6RK",
    "B0CNL9VZSR",
    "B0CNLVQBJ4",
    "B07SKLLYTW",
    "B0BWPK91TB",
    "B0BX7K85PK",
    "B0BX7KWG6F",
    "B0CP9YB3Q4",
    "B0BHBVWJWX"
],
        "zip_format": r"^[ABCEGHJKLMNPRSTVXY]\d[A-Z] \d[A-Z]\d$",  # 加拿大邮编格式（字母+数字组合）
        "zip_list": [
            'M5V2T6',
            'H3Z2Y7',
            'V6B1B7',
            'B3H2Y5',
            'K1A0B1',
            'T2H1Z3',
            'V6B1B7',
            'A0A0A0',
            'G1A1A1',
            'L4W1A1',
            'K1A0A6',
            'K1A0H3',
            'M5S1A1',
            'V6T1Z4',
            'M5B2H1',
            'V5H4M1',
            'V5Y1V4',
            'H2Y1C6',
            'M5J1E6',
            'K1P5G4',
            'M5H2N2',
            'M5V3H1',
            'H3A0G4',
            'V5A1S6',
            'J8X4B7',
            'V6Z3B7',
            'M5J2P1',
            'V6A4C1',
            'V6C3T4',
            'V6C0C3'

        ],
        "locators": {
            "zip_input_0": (By.ID, "GLUXZipUpdateInput_0"),
            "zip_input_1": (By.ID, "GLUXZipUpdateInput_1"),
            "LOCATORS_CA": [
    (By.ID, "GLUXZipUpdate"),  # 最直接的按钮容器ID
    (By.XPATH, "//input[@type='submit' and contains(@aria-labelledby, 'GLUXZipUpdate-announce')]"),
    (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Apply']/ancestor::span[@class='a-button-inner']/input"),
    (By.XPATH, "//span[text()='Apply']/ancestor::span[@id='GLUXZipUpdate']//input")
]
        },
        "success_text": [
            "We will use your selected location to show all products available for the",
            "我们将使用您选择的位置显示所有可用的产品"
        ]
    },
    "uk": {  # 新增英国站配置
        "country": "英国",
        "lang": "en-GB",
        "product_url": "https://www.amazon.co.uk/dp/{asin}/",
        "asin_list": ["B0C9VVCL12", "B0DR67SQ8N", "B088W5HWVX", "B0CW18TFWZ","B0CJ63CKFY", "B08MFMSKJF"],  # 示例ASIN
        "zip_format": r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$",  # 英国邮编格式（如SW1A 1AA）
        "zip_list": ['SW1A1AA',
                    'M11AE',
                    'B11BB',
                    'EH13QR',
                    'CB21TN',
                    'OX12JD',
                    'BS15TR',
                    'L31AH',
                    'LS11UR',
                    'G21DU',
                    'NG15FS',
                    'S12HH',
                    'NE14LP',
                    'CF101EP',
                    'BT15GS',
                    'AB101XG',
                    'ZE10AA',
                    'IV11HT',
                    'PO13AX',
                    'SO147DW',
                    'TA11AA',
                    'PL11DH',
                    'TR12XQ',
                    'DD11NL',
                    'DN12HF',
                    'SR11RH',
                    'LA11AA',
                    'YO17HH',
                    'WN11XX'  ],
        "locators": {
            "zip_input": (By.ID, "GLUXZipUpdateInput"),  # 假设英国站输入框ID与美国站类似
            "LOCATORS_UK": [
                (By.XPATH, "//span[@id='GLUXZipUpdate']//input[@type='submit']"),
                (By.XPATH,
                 "//div[@class='a-column a-span4 a-span-last']//span[@class='a-button a-button-span12']//input[@type='submit']"),
                (By.XPATH,
                 "//input[contains(@class, 'a-button-input') and @type='submit']") # 保留原始定位器
]
            },
        "success_text": [
            "We'll use this address to show you products available in your area",
            "我们将使用此地址向您显示您所在地区的可用产品"
        ]
    },
    "fr": {  # 新增英国站配置
            "country": "法国",
            "lang": "fr-FR",
            "product_url": "https://www.amazon.fr/dp/{asin}/",
            "asin_list": ["B0DQ5NTHB8", "B0F3VX3BWJ", "B0DVGDNNXM", "B0F447GZ6H","B0895LY6F1", "B07YF46MRY"],  # 示例ASIN
            "zip_format": r'^\d{5}$',  # 英国邮编格式（如SW1A 1AA）
            "zip_list": ['75001',
                        '13001',
                        '69001',
                        '06000',
                        '31000',
                        '44000',
                        '33000',
                        '67000',
                        '59000',
                        '72000',
                        '51100',
                        '37000',
                        '42000',
                        '80000',
                        '34000',
                        '76000',
                        '57000',
                        '14000',
                        '64000',
                        '87000'],
            "locators": {
                "zip_input": (By.ID, "GLUXZipUpdateInput"),  # 假设英国站输入框ID与美国站类似
                "LOCATORS_FR": [
                            #方法1：通过 aria-labelledby 精准匹配 submit 按钮（最推荐）
                            (By.XPATH, "//input[@type='submit' and contains(@class, 'a-button-input') and @aria-labelledby='GLUXZipUpdate-announce']"),

                            # 方法2：通过按钮文字 'Actualiser' 向上回溯找到对应 input（文字匹配，次稳）
                            (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Actualiser']/preceding-sibling::input[@type='submit']"),

                            # 方法3：匹配整个按钮容器 <span>（如需点击按钮包裹层而不是 input）
                            (By.XPATH, "//span[@data-action='GLUXPostalUpdateAction']//input[@type='submit']"),

                            # 方法4：点击整个按钮文本容器（如果 input 不可点击）
                            (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Actualiser']")
                        ]



                },
            "success_text": [
                "We'll use this address to show you products available in your area",
                "我们将使用此地址向您显示您所在地区的可用产品"
            ]
        },
    "de": {  # 德国站配置
        "country": "德国",
        "lang": "de-DE",  # 修改为德语
        "product_url": "https://www.amazon.de/dp/{asin}/",  # URL改为德国站点
        "asin_list": ["B09379LLZ7", "B07864Y4DG", "B0DT1DHP5S", "B0C3M5MS3N","B07BMZYYFZ", "B0F3NMVFPK"],
        "zip_format": r'^\d{5}$',  # 德国邮编也是5位数字
        "zip_list": ['10115',  # 柏林中心
                    '80331',  # 慕尼黑中心
                    '50667',  # 科隆老城
                    '20095',  # 汉堡市中心
                    '60311',  # 法兰克福市中心
                    '70173',  # 斯图加特市中心
                    '01067',  # 德累斯顿老城
                    '04109',  # 莱比锡市中心
                    '19053',  # 什未林
                    '40213',  # 杜塞尔多夫市中心
                    '01097',  # 德累斯顿新城
                    '50677',  # 科隆南城
                    '80686',  # 慕尼黑西区
                    '22087',  # 汉堡哈芬城
                    '60329',  # 法兰克福火车总站
                    '79100',  # 弗莱堡
                    '66111',  # 萨尔布吕肯
                    '53173',  # 波恩巴特戈德斯贝格
                    '01069',  # 德累斯顿中心火车站
                    '40210'   # 杜塞尔多夫媒体港
        ],
        "locators": {
            "zip_input": (By.ID, "GLUXZipUpdateInput"),  # 通常相同

            # 修改定位器使用德语界面元素
            "LOCATORS_DE": [
                # 方法1: 使用德语按钮文本 "Übernehmen" (应用)
                (By.XPATH, "//input[@type='submit' and @aria-labelledby='GLUXZipUpdate-announce']"),
                (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Bestätigen']//ancestor::span[contains(@class, 'a-button-inner')]"),

                # 方法2: 通用结构匹配
                (By.XPATH, "//div[@class='a-column a-span4 a-span-last']//span[@data-action='GLUXPostalUpdateAction']//input[@type='submit']"),

            ]
        },
        "success_text": [
            "Wir verwenden diese Adresse, um Ihnen Produkte anzuzeigen",  # 德语原文
            "show you products available in your area"  # 英语备用
        ]
    },
    "it": {  # 意大利站配置
        "country": "意大利",
        "lang": "it-IT",
        "product_url": "https://www.amazon.it/dp/{asin}/",
        "asin_list": ["B00DRDXN38", "B07RSCPH4N", "B00G5YOVZA", "B0D1RHN5WW", "B07SPWGN1R"],  # 示例 ASIN，可换
        "zip_format": r'^\d{5}$',
        "zip_list": [
            "00184",  # 罗马
            "20121",  # 米兰
            "30124",  # 威尼斯
            "40121",  # 博洛尼亚
            "50123",  # 佛罗伦萨
            "80132",  # 那不勒斯
            "09124",  # 卡利亚里
            "65121",  # 佩斯卡拉
            "16121",  # 热那亚
            "95131",  # 卡塔尼亚
            "35121",  # 帕多瓦
            "61121",  # 佩萨罗
            "70121",  # 巴里
            "98122",  # 墨西拿
            "89125",  # 雷焦卡拉布里亚
            "72100",  # 布林迪西
            "90133",  # 巴勒莫
            "71121",  # 福贾
            "44121",  # 费拉拉
            "81100"   # 卡塞塔
        ],
        "locators": {
            "zip_input": (By.ID, "GLUXZipUpdateInput"),
            "LOCATORS_IT": [
            # 方法1：根据按钮 ID 精确定位（最稳妥）
            (By.ID, "GLUXZipUpdate"),

            # 方法2：根据按钮文字 "Conferma" 匹配
            (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Conferma']//ancestor::span[contains(@class, 'a-button-inner')]"),

            # 方法3：结构通用匹配（备用方案）
            (By.XPATH, "//span[@data-action='GLUXPostalUpdateAction']//input[@type='submit']")
        ]

        },
        "success_text": [
            "Utilizziamo questo indirizzo per mostrarti i prodotti disponibili nella tua zona",  # 意大利语提示
            "show you products available in your area"  # 英语备用
        ]
    },
    "es": {  # 西班牙站配置
        "country": "西班牙",
        "lang": "es-ES",
        "product_url": "https://www.amazon.es/dp/{asin}/",
        "asin_list": ["B0F8BQ3PCL", "B0F8BLYRR3", "B0FBMD83DV", "B0DSWFHTL2", "B0D7V1R6VM", "B0CB12CPCY"],
        "zip_format": r'^\d{5}$',
        "zip_list": [
                "28001",  # 马德里
                "08001",  # 巴塞罗那
                "41001",  # 塞维利亚
                "46001",  # 瓦伦西亚
                "29001",  # 马拉加
                "35001",  # 拉斯帕尔马斯
                "50001",  # 萨拉戈萨
                "07001",  # 帕尔马
                "03001",  # 阿利坎特
                "15001",  # 拉科鲁尼亚
                "48001",  # 毕尔巴鄂
                "24001",  # 莱昂
                "26001",  # 洛格罗尼奥
                "33001",  # 奥维耶多
                "10001",  # 卡塞雷斯
                "37001",  # 萨拉曼卡
                "31001",  # 潘普洛纳
                "20001",  # 圣塞瓦斯蒂安
                "18001",  # 格拉纳达
                "02001"   # 阿尔瓦塞特
            ],
        "locators": {
            "zip_input": (By.ID, "GLUXZipUpdateInput"),
            "LOCATORS_ES": [
                # 方法1：根据按钮 ID 精确定位（推荐，最稳定）
                (By.ID, "GLUXZipUpdate"),

                # 方法2：匹配按钮文字“Confirmar”
                (By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Confirmar']//ancestor::span[contains(@class, 'a-button-inner')]"),

                # 方法3：结构通用匹配（备用）
                (By.XPATH, "//span[@data-action='GLUXPostalUpdateAction']//input[@type='submit']")
            ]

        },
        "success_text": [
            "Usamos esta dirección para mostrarte productos disponibles en tu zona",  # 西班牙语提示
            "show you products available in your area"
        ]
    }

}


class ProxyManager:
    def __init__(self):
        self.primary_proxy = 0  # 当前使用的主代理索引
        self.proxy_group = [
            'http:',  # 快代理1主
            'http:',  # 快代理1备
        ]
        self.last_group_time = datetime.min  # 上次使用组的时间戳，初始化都为0
        self.change_ip = ""
        # self.change_ip = ""
        self.change_ip_num = 0
        # 状态跟踪
        self.current_proxy = None  # 当前使用的完整代理地址

    def get_proxy(self):
        return self.get_group_proxy()

    def get_group_proxy(self):
        proxy = self.proxy_group[self.primary_proxy]
        self.last_group_time = datetime.now()
        self.current_proxy = proxy
        return proxy

    def _is_refresh_group_proxy(self):
        if not self.last_group_time:
            return True
        if ((datetime.now() - self.last_group_time).total_seconds()) >= 1800:  # 30分钟
            return True
        return False
    def is_change_ip_num(self):
        if not self.last_group_time:
            return True
        if ((datetime.now() - self.last_group_time).total_seconds()) >= 600:  # 10分钟
            return True
        return False
    def to_change_ip_num(self):
        if self.is_change_ip_num():
            self.change_ip_num = 0

    def is_change_ip(self):
        self.to_change_ip_num()
        if self.change_ip_num < 2:
            return True
        return False

    def to_change_ip(self):
        if self.is_change_ip():
            response = requests.get(self.change_ip)
            logging.info(f"改变IP:{response.text}")
        else:
            logging.info(f"10分钟内修改IP次数用尽，需要等待")



    def check_primary_proxy(self):
        # self.primary_proxy = (self.primary_proxy + 1) % 4
        self.primary_proxy = (self.primary_proxy + 1) % 2

class AmazonScraper:
    def __init__(self, proxy_manager, nation):
        self.driver = None
        self.proxy_url = None
        self.is_ready = False
        self.proxy_manager = proxy_manager
        self.nation = nation
        self.site = nation  # 如 'us', 'ca', 'uk'
        self.site_config = SITE_CONFIGS.get(nation)
        if not self.site_config:
            raise ValueError(f"Unsupported site: {nation}")

        self.product_url_template = self.site_config["product_url"]
        self.language = self.site_config["lang"]
        self.asin_list = self.site_config["asin_list"]
        self.zip_format = re.compile(self.site_config["zip_format"])
        self.zip_list = self.site_config["zip_list"]
        self.locators = self.site_config["locators"]
        self.success_text_patterns = self.site_config["success_text"]
        # 模拟用户行为
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:116.0) Gecko/20100101 Firefox/116.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Whale/3.23.214.10 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
        ]
        # 生成现代浏览器UA（避免使用旧版本）
        self.ua = UserAgent(
            browsers=['chrome', 'firefox', 'safari'],  # 限定浏览器类型
            min_percentage=1.3  # 只使用市场占有率>1.3%的UA
        )
        # 分辨率
        self.resolutions = [
            (1366, 768), (1920, 1080),
            (1600, 900), (1280, 720),
        ]



        # 反爬虫JavaScript
        self.js_script = """
                                                    const originalGetContext = HTMLCanvasElement.prototype.getContext;
                                                    HTMLCanvasElement.prototype.getContext = function(...args) {
                                                        if (args[0] === '2d') {
                                                            const ctx = originalGetContext.apply(this, args);
                                                            ctx.__canvas2d_fingerprint__ = "randomized";  // 干扰指纹
                                                            return ctx;
                                                        }
                                                        return originalGetContext.apply(this, args);
                                                    };
                                                    """

        # 代理验证网站
        self.validate_url = [ ]

        # 开始初始化
        while not self.is_ready:
            try:
                init_driver_s = time.time()
                try:
                    # 初始化driver
                    self.driver = self._init_driver()
                except:
                    # TODO 有可能一连串代理IP不健康，可以适当等一会，不要请求频繁，这样一连串也是不健康的
                    time.sleep(5)
                    continue
                init_driver_e = time.time()
                logging.info(f"driver加载成功完成耗时:{init_driver_e - init_driver_s}s")
                self.ocr = ddddocr.DdddOcr()
                #print('1111111')
                # 获取邮编
                self.current_zip = self._get_random_zip()
                self.last_verified_zip = None
                self.retry_count = 0
                try:
                    if self._detect_throttling(self.driver):
                        self._recover_from_throttling()
                    #print('-------------')
                    # 新增初始化邮编设置

                    if not self._setup_initial_zip():
                        raise RuntimeError("Initial ZIP code setup failed")

                    #print('========')
                    self.is_ready = True
                    logging.info(f" 浏览器初始设置成功")
                except:
                    logging.info(f"初始化失败，重试")
                    time.sleep(2)
            except Exception as e:
                if self.driver:
                    #print('222222222222')
                    try:
                        self.driver.close()
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                logging.info(f"初始化失败: {str(e)}")
                logging.info(f"�� 等待 5 秒后重试...")
                self.proxy_manager.check_primary_proxy()
                time.sleep(5)  # 等待一段时间后重试

            except KeyboardInterrupt:
                logging.info(f"用户自己退出")
                raise

    def _init_driver(self):
        """driver 初始化尝试 3次"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logging.info(f"⚠ 开始初始driver尝试{attempt}/{max_retries}")
            # 每次driver 显示声明
            driver = None
            options = webdriver.ChromeOptions()
            # try:
            #     # selected_ua = self.ua.chrome if random.choice([True, False]) else self.ua.firefox
            #     # TODO 先暂时只有chrome
            #     selected_ua = self.ua.chrome
            #     logging.info(f"生成的user_agent为: {str(selected_ua)}")
            # except:
            #     selected_ua = random.choice(self.user_agents)  # 随机用户代理
            #     logging.info(f"使用的列表中的随机user_agent为: {str(selected_ua)}")
            selected_ua = random.choice(self.user_agents)  # 随机用户代理
            logging.info(f"使用的列表中的随机user_agent为: {str(selected_ua)}")
            width, height = random.choice(self.resolutions)
            prefs = {
                "profile.managed_default_content_settings.images": 2,  # 禁用图片
                "profile.default_content_setting_values.media": 2,
                # "profile.managed_default_content_settings.stylesheets": 2,  # 禁用 CSS
                # "profile.managed_default_content_settings.javascript": 2,  # 禁用 JavaScript
            }
            # 请求代理IP
            self.proxy_ip = self.proxy_manager.get_proxy()
            options.add_argument(f"--proxy-server={self.proxy_ip}")

            # 浏览器配置
            if self.site == "uk":
                options.add_argument("--start-maximized")  # 英国站可能需要全屏
            chrome_binary = r"path\to\chrome.exe"
            options.binary_location = chrome_binary
            options.add_experimental_option("prefs", prefs)  # 禁用非必要加载项
            options.add_argument(f"--window-size={width},{height}")
            options.add_argument(f"--force-device-scale-factor={random.uniform(0.8, 1.2):.2f}")  # 随机缩放比例
            options.add_argument(f"--lang={self.language}")
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-notifications')
            options.add_argument('--disable-popup-blocking')
            options.add_argument('--disable-web-security')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--no-sandbox')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            options.page_load_strategy = 'eager'
            options.add_argument(f"user-agent={selected_ua}")  # 只保留这一行
            #options.add_argument("--headless")  # 可选，无头模式

            executable_path = r"C:\Users\dell\.wdm\drivers\chromedriver\win64\137.0.7151.55\chromedriver-win32/chromedriver.exe"
            service = Service(
                executable_path=executable_path
            )
            try:
                driver = webdriver.Chrome(service=service, options=options)
                logging.info(f"driver session_id:{driver.session_id},站点[{self.site}]浏览器初始化成功")
            except (TimeoutException or Exception) as e:
                logging.error(f"driver session 加载时长过长, 换session")
                if driver:
                    driver.close()
                    driver.quit()
                continue

            if not self.validate_proxy(driver):
                logging.info(f"代理不健康")
                if driver:
                    driver.close()
                    driver.quit()
                    logging.info(" 清理当前浏览器实例")
                time.sleep(2)
                self.proxy_manager.check_primary_proxy()
                continue


            logging.info(f"driver实例初始化完成")
            return driver
        logging.error(f" 初始化最终失败")
        raise RuntimeError(f"连续{max_retries}次初始化失败")

    def validate_proxy(self, driver):
        for url in self.validate_url:
            print(url)
            try:
                try:
                    driver.get(url)
                    logging.info(f"打开验证网站成功")
                    print(driver.page_source)
                except Exception as e:
                    logging.info(f"打开验证网站失败:{str(e)}")
                    continue
                try:
                    soup_str = BeautifulSoup(driver.page_source, "html.parser")
                    # pre_content = soup_str.find("pre").text.strip()  # 提取 pre 标签内的文本
                    # # 用正则表达式匹配IP地址
                    # ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', pre_content)
                    # if ip_match:
                    #     logging.info(f"正确使用代理IP:{ip_match.group()}")
                    #     return True
                    pre_tag = soup_str.find("pre")
                    if pre_tag:
                        pre_content = pre_tag.text.strip()
                        ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', pre_content)
                        if ip_match:
                            logging.info(f"正确使用代理IP:{ip_match.group()}")
                            return True
                    else:
                        logging.info("页面中未找到 <pre> 标签")
                except Exception as e:
                    logging.info(f"验证代理失败:{str(e)}")
                    continue
            except:
                continue

        return False


    def _setup_initial_zip(self):
        """初始化时设置邮编（只执行一次）"""
        max_retries = 2  # 减少重试次数避免过多刷新
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f" 初始化邮编设置尝试 #{attempt}")

                if self.change_delivery_address(self.current_zip):
                    logging.info(f"✅ 初始邮编 {self.current_zip} 设置成功")
                    return True
            except Exception as e:
                logging.error(f"初始化邮编设置失败: {str(e)}")
                # 发生异常时重新加载原始页面
                asin = random.choice(self.site_config['asin_list'])  # 需要站点配置包含asin列表
                product_url = self.product_url_template.format(asin=asin)
                self.get_page(product_url)
        return False

        # TODO
        # 随机从数据库抽一个ASIN出来打开，需要单独新建一个表，我们手动往这个表插没有A+页面的ASIN

    def is_modal_valid_us(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def is_modal_valid_it(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def is_modal_valid_es(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def is_modal_valid_ca(self):
        try:
            # 关键元素存在性检测
            print(self.locators.get("zip_input_0"))
            input1_displayed = self.driver.find_element(*self.locators.get("zip_input_0")).is_displayed()
            modal1_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")

            input2_displayed = self.driver.find_element(*self.locators.get("zip_input_1")).is_displayed()

            return all([input1_displayed, modal1_visible, input2_displayed, True])
        except Exception as e:
            print(f"[is_modal_valid_ca] error: {e}")
            return False

    def is_modal_valid_uk(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def is_modal_valid_fr(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def is_modal_valid_de(self):
        try:
            # 关键元素存在性检测
            input_displayed = self.driver.find_element(*self.locators.get("zip_input")).is_displayed()
            modal_visible = "visible" in self.driver.find_element(
                By.CSS_SELECTOR, "div.a-popover-modal"
            ).get_attribute("style")
            return all([input_displayed, modal_visible, True])
        except:
            return False

    def change_delivery_address(self, zip_code=None):
        if True:
            asin = random.choice(self.site_config['asin_list'])  # 需要站点配置包含asin列表
            product_url = self.product_url_template.format(asin=asin)
            self.get_page(product_url)
            # 等待必要元素
            wait = WebDriverWait(self.driver, 10)
            wait.until(
                EC.presence_of_element_located((By.ID, 'nav-global-location-popover-link'))
            )
            logging.info(f"邮编图标加载成功")
            wait.until(
                EC.presence_of_element_located((By.ID, 'nav-packard-glow-loc-icon'))
            )
            logging.info(f"邮编按钮加载成功")
            self.driver.execute_script("window.stop();")

            if not self.wait_for_page_loaded(self.driver):
                raise RuntimeError("页面加载未完成")
            logging.info(f"页面加载成功")

        if hasattr(self, '_zip_initialized') and self._zip_initialized:
            logging.warning("邮编初始化已完成，跳过后续修改")
            return True

        target_zip = zip_code or self._get_random_zip()
        max_retries = 5
        success = False


        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f" 初始化邮编设置尝试 #{attempt}")
                logging.info(f" 尝试#{attempt} 修改邮编至 {target_zip}")

                # 点击地址图标
                address_icon = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "nav-global-location-popover-link"))
                )
                try:
                    # 点击
                    self.driver.execute_script("arguments[0].click();", address_icon)
                    logging.info(f" 常规点击邮编图标成功")
                    zip_input = WebDriverWait(self.driver, 30).until(
                        EC.element_to_be_clickable((By.XPATH, "//*[contains(@id, 'ZipUpdate')][@type='text']")))
                    self.driver.execute_script("arguments[0].click();", zip_input)
                    logging.info(f"点击邮编输入框成功")
                    self.driver.execute_script("window.stop();")
                except TimeoutException:
                    try:
                        self.driver.execute_script("arguments[0].click();", address_icon)
                        zip_input = WebDriverWait(self.driver, 30).until(
                            EC.element_to_be_clickable((By.ID, "nav-global-location-data-modal-action")))
                        logging.info(f" Javascript点击邮编图标成功")
                    except Exception as e:
                        try:
                            address_icon = WebDriverWait(self.driver, 10).until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, "//*[contains(text(),'Update location') or contains(text(),'更改此地址') ]"))
                            )
                            self.driver.execute_script("arguments[0].click();", address_icon)
                            zip_input = WebDriverWait(self.driver, 30).until(
                                EC.element_to_be_clickable((By.ID, "nav-global-location-data-modal-action")))
                        except:
                            logging.info(f" 3种点击都失效")
                # 等待邮编输入框出现
                zip_input = WebDriverWait(self.driver, 30).until(
                    EC.element_to_be_clickable((By.ID, "nav-global-location-data-modal-action"))
                )
                if not zip_input:
                    for _ in range(3):
                        self.driver.execute_script("arguments[0].click();", address_icon)
                        zip_input = WebDriverWait(self.driver, 30).until(
                            EC.element_to_be_clickable((By.ID, "nav-global-location-popover-link"))
                        )
                        logging.info(f" 点击邮编图标成功")
                        if zip_input:
                            break
                logging.info(f" 找到邮编输入框按钮")

                # 输入邮编前验证弹窗
                time.sleep(2)
                if self.nation == 'us':
                    if not self.is_modal_valid_us():
                        raise Exception("弹窗在输入前已失效")
                    self._input_us_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_us(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_US'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_us(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                           try {
                                               document.querySelector('button[data-action=a-popover-close]').click();
                                               window.location.reload();
                                           } catch(e){}
                                       """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'ca':
                    if not self.is_modal_valid_ca():
                        raise Exception("弹窗在输入前已失效")

                    self._input_ca_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_ca(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_CA'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_ca(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                                               try {
                                                                   document.querySelector('button[data-action=a-popover-close]').click();
                                                                   window.location.reload();
                                                               } catch(e){}
                                                           """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'uk':
                    if not self.is_modal_valid_uk():
                        raise Exception("弹窗在输入前已失效")
                    self._input_uk_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_uk(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_UK'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_uk(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                                               try {
                                                                   document.querySelector('button[data-action=a-popover-close]').click();
                                                                   window.location.reload();
                                                               } catch(e){}
                                                           """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'fr':
                    if not self.is_modal_valid_fr():
                        raise Exception("弹窗在输入前已失效")
                    self._input_fr_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_fr(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_FR'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_fr(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"⚠️ 未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                           try {
                                               document.querySelector('button[data-action=a-popover-close]').click();
                                               window.location.reload();
                                           } catch(e){}
                                       """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'de':
                    if not self.is_modal_valid_de():
                        raise Exception("弹窗在输入前已失效")
                    self._input_de_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_de(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_DE'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_de(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                           try {
                                               document.querySelector('button[data-action=a-popover-close]').click();
                                               window.location.reload();
                                           } catch(e){}
                                       """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'it':
                    if not self.is_modal_valid_it():
                        raise Exception("弹窗在输入前已失效")
                    self._input_it_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_it(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_IT'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_it(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                           try {
                                               document.querySelector('button[data-action=a-popover-close]').click();
                                               window.location.reload();
                                           } catch(e){}
                                       """)
                        raise Exception("邮编提交状态验证失败")
                elif self.nation == 'es':
                    if not self.is_modal_valid_us():
                        raise Exception("弹窗在输入前已失效")
                    self._input_es_zip_code(target_zip, attempt)

                    WebDriverWait(self.driver, 3).until(
                        lambda d: self.is_modal_valid_us(),
                        "输入邮编后弹窗失效"
                    )
                    # ========== 修改点：多定位器轮询 ==========
                    submit_button = None
                    for idx, locator in enumerate(self.locators.get('LOCATORS_ES'), 1):
                        try:
                            submit_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable(locator)
                            )
                            logging.info(f"定位器#{idx} 成功找到设置按钮")
                            break
                        except Exception as e:
                            logging.warning(f"定位器#{idx} 失败: {str(e)}")
                            # self._capture_debug_snapshot(f"locator_{idx}_fail")
                    if not submit_button:
                        self._manual_debug_helper(None)
                        raise Exception("所有定位器均无法找到设置按钮")

                    # ========== 增强点击验证流程 ==========
                    click_success = self._retry_click_with_validation_es(submit_button, target_zip)

                    if not click_success:
                        # self._manual_debug_helper(submit_button)
                        # raise Exception("所有点击方式均失败，请查看手动调试窗口")
                        logging.info(f"所有点击方式均失败")
                        self.driver.refresh()
                        continue

                    try:

                        # 智能页面刷新
                        logging.info(" 等待页面稳定...")
                        # if self._verify_zip_change(target_zip):
                        #     success = True
                        #     break
                        success = True
                        break
                    except Exception as e:
                        logging.warning(f"未检测到成功状态: {str(e)}")
                        self.driver.execute_script("""
                                           try {
                                               document.querySelector('button[data-action=a-popover-close]').click();
                                               window.location.reload();
                                           } catch(e){}
                                       """)
                        raise Exception("邮编提交状态验证失败")


                # ========== 成功状态检测 ==========

            except Exception as e:
                logging.error(f"尝试#{attempt} 失败: {str(e)}")
                # self._capture_debug_snapshot(f"attempt_{attempt}_fail")
                if "弹窗" in str(e) or "modal" in str(e).lower():
                    self.driver.refresh()
                    time.sleep(2)
        logging.info("正在等待邮编变更生效...")
        time.sleep(2)
        if success:
            self.last_verified_zip = target_zip
            self.current_zip = target_zip
            self._zip_initialized = True
        return success

    def get_page(self, url):
        try:
            self.driver.get(url)
            if self._is_captcha_required():
                logging.info("检测到验证码，开始处理...")
                # result = self.solve_captcha()
                result = self.solve_captcha_new()
                status_msg = "已通过验证码" if result else "‼️ 验证码处理失败"
                logging.info(status_msg)
            else:
                logging.info(" 当前无验证码")
                status_msg = " 未遇到验证码"
                result = True
        except TimeoutException as e:
            # 新增验证码检测逻辑
            if self._is_captcha_required():
                logging.info("检测到验证码，开始处理...")
                # result = self.solve_captcha()
                result = self.solve_captcha_new()
                status_msg = "已通过验证码" if result else "‼️ 验证码处理失败"
                logging.info(status_msg)
            else:
                logging.info(" 当前无验证码")
                status_msg = " 未遇到验证码"
                result = True
            # 等待核心元素
            wait = WebDriverWait(self.driver, 25)
            wait.until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            logging.info(f"大框架加载成功")
            wait.until(
                EC.presence_of_element_located((By.ID, 'twotabsearchtextbox'))
            )
            logging.info(f"搜素框加载成功")
            wait.until(
                EC.presence_of_element_located((By.ID, 'nav-search-submit-button'))
            )
            logging.info(f"搜索按钮加载成功")
            wait.until(
                EC.presence_of_element_located((By.ID, 'nav-global-location-popover-link'))
            )
            logging.info(f"邮编图标加载成功")
            wait.until(
                EC.presence_of_element_located((By.ID, 'nav-packard-glow-loc-icon'))
            )
            logging.info(f"邮编按钮加载成功")

            self.driver.execute_script("window.stop();")
        # 新增人类行为模拟
        try:
            # 初始随机移动
            self.random_mouse_move()

            # 滚动后再次移动鼠标
            self.random_mouse_move()

        except Exception as e:
            logging.warning(f"人类行为模拟失败: {str(e)}")
        return result

    def _is_captcha_required(self):
        """更精准的验证码检测"""
        captcha_indicators = [
            "Type the characters you see in this image:",  # 英文
            "输入图中字符",  # 中文
            "captchacharacters",  # 输入框ID
            "image-captcha-section"  # 图片容器class
        ]
        return any(indicator in self.driver.page_source for indicator in captcha_indicators)

    def solve_captcha(self):
        return self.solve_captcha_free()

    def solve_captcha_new(self):
        captcha_text = self.get_captcha_text()
        logging.info(f"captcha_text:{captcha_text}")
        if captcha_text is None or captcha_text == "" or captcha_text == "Not solved":
            return False
            # 模拟人工输入
        input_box = self.driver.find_element(By.ID, "captchacharacters")
        for char in captcha_text:
            input_box.send_keys(char)
            time.sleep(random.uniform(0.1, 0.3))

        time.sleep(random.uniform(0.5, 1.2))
        return self._submit_captcha(captcha_text)

    def get_captcha_text(self):
        try:
            # 1.使用amazoncaptcha
            # 1.1，直接使用driver获取验证码
            try:
                time.sleep(1)
                captcha = AmazonCaptcha.fromdriver(self.driver)
                captcha_text = captcha.solve()  # 识别后返回的结果，字符型
                if captcha_text is not None and captcha_text != "" and captcha_text != 'Not solved':
                    return captcha_text
            except:
                logging.info(f"amazoncaptcha直接使用driver提取验证码失败")

            # 1.2 使用图片链接
            try:
                captcha_img = self.driver.find_element(By.XPATH, "//div[contains(@class,'a-text-center')]/img")
                img_src = captcha_img.get_attribute("src")
                captcha = AmazonCaptcha.fromlink(img_src)
                captcha_text = captcha.solve()  # 识别后返回的结果，字符型
                if captcha_text is not None and captcha_text != "" and captcha_text != 'Not solved':
                    return captcha_text
            except:
                logging.info(f"amazoncaptcha直接使用link提取验证码失败")

            # 2.使用ddddocr识别（直接传入bytes）
            try:
                captcha_img = self.driver.find_element(By.XPATH, "//div[contains(@class,'a-text-center')]/img")
                img_src = captcha_img.get_attribute("src")
                if img_src.startswith("http"):
                    response = requests.get(img_src, timeout=10)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    raise ValueError("Unsupported image source format")

                # 使用ddddocr识别（直接传入bytes）
                captcha_text = self.ocr.classification(image_bytes)
                if captcha_text is not None and captcha_text != "" and captcha_text != 'Not solved':
                    return captcha_text
            except requests.exceptions.RequestException as e:
                logging.error(f"ddddocr图片下载失败: {str(e)}")
                return None
        except Exception as e:
            logging.error(f"验证码处理异常: {str(e)}", exc_info=True)
            return None

    def solve_captcha_free(self):
        try:
            captcha_img = self.driver.find_element(By.XPATH, "//div[contains(@class,'a-text-center')]/img")
            img_src = captcha_img.get_attribute("src")

            # # 处理base64格式
            # if img_src.startswith("data:image"):
            #     if "base64" not in img_src:
            #         raise ValueError("Unsupported image encoding")
            #     base64_data = img_src.split("base64,")[1]
            #     image_bytes = base64.b64decode(base64_data)
            # 处理普通URL格式
            if img_src.startswith("http"):
                response = requests.get(img_src, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            else:
                raise ValueError("Unsupported image source format")

            # 使用ddddocr识别（直接传入bytes）
            captcha_text = self.ocr.classification(image_bytes)

            # 模拟人工输入
            input_box = self.driver.find_element(By.ID, "captchacharacters")
            for char in captcha_text:
                input_box.send_keys(char)
                time.sleep(random.uniform(0.1, 0.3))

            time.sleep(random.uniform(0.5, 1.2))
            return self._submit_captcha(captcha_text)

        except requests.exceptions.RequestException as e:
            logging.error(f"图片下载失败: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"验证码处理异常: {str(e)}", exc_info=True)
            return False

    def _wait_element(self, xpath, timeout=10):
        """显式等待元素存在"""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
        except TimeoutException:
            logging.error(f"元素定位超时: {xpath}")
            raise

    def is_captcha_passed(self):
        """多重验证机制"""
        checks = [
            lambda: "Type the characters" not in self.driver.page_source,
            lambda: self.driver.find_elements(By.ID, "captchacharacters") == [],
            lambda: "Sorry" not in self.driver.page_source
        ]
        return all(check() for check in checks)

    def _submit_captcha(self, captcha_text):
        """提交验证码"""
        try:
            # captcha_input = self.driver.find_element(By.ID, "captchacharacters")
            # captcha_input.clear()
            # captcha_input.send_keys(captcha_text)
            # captcha_input.send_keys(Keys.RETURN)
            captcha_confrim_butten = self.driver.find_element(By.XPATH, '//button[@type="submit"]')
            captcha_confrim_butten.click()
            self._wait_element('//*[@id="twotabsearchtextbox"]')
            # 等待验证码处理结束

            # 验证验证码是否通过
            if self.is_captcha_passed():
                logging.info("验证码通过")
                return True
            else:
                logging.error("验证码未通过")
                return False
        except Exception as e:
            logging.error(f"提交验证码异常: {str(e)}")
            return False

    def random_mouse_move(self):
        """无参数快速随机鼠标移动（执行时间<0.5秒）"""
        driver = self.driver
        action = ActionChains(driver)

        # 获取当前窗口尺寸
        width = driver.execute_script("return window.innerWidth")
        height = driver.execute_script("return window.innerHeight")

        # 生成3-5个随机路径点（包含抖动）
        points = []
        for _ in range(random.randint(3, 5)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            # 添加微小抖动
            x += random.randint(-3, 3)
            y += random.randint(-3, 3)
            points.append((max(0, min(x, width)), max(0, min(y, height))))

        # 计算分段延迟（确保总时间<0.5秒）
        segment_delay = 0.5 / len(points)

        # 执行快速移动
        for x, y in points:
            driver.execute_script(f"window.moveTo({x}, {y});")
            time.sleep(segment_delay * random.uniform(0.7, 1.3))  # 随机变速

        # 添加最终微小抖动（模拟停留时的手部自然抖动）
        for _ in range(2):
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            driver.execute_script(f"window.moveBy({dx}, {dy});")
            time.sleep(0.02)

    def wait_for_page_loaded(self, driver, timeout=30):
        """接收driver参数"""
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState === 'complete'")
            )
            # 其他验证条件...
            return True
        except TimeoutException:
            return False

    def _input_us_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_us_zip_code(zip_code, attempt + 1)

            raise

    def _input_es_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_us_zip_code(zip_code, attempt + 1)

            raise

    def _input_it_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_us_zip_code(zip_code, attempt + 1)

            raise

    def _input_ca_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input1 = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.ID, "GLUXZipUpdateInput_0")),
                "邮编输入框定位超时"
            )

            zip_input2 = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.ID, "GLUXZipUpdateInput_1")),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                for zip_input in [zip_input1, zip_input2]:
                    zip_input.send_keys(Keys.CONTROL + 'a')
                    zip_input.send_keys(Keys.DELETE)
                    self.driver.execute_script("arguments[0].value = '';", zip_input)
                    time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input1 = document.getElementById('GLUXZipUpdateInput_0');
                const input2 = document.getElementById('GLUXZipUpdateInput_1');
                input1.value = '{zip_code[:3]}';
                input2.value = '{zip_code[3:]}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input1.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                    input2.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input1.value),
                    GLUX.validatePostalCode(input2.value);
                    
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            # 修改验证逻辑，同时检查两个输入框
            WebDriverWait(self.driver, 5).until(
                lambda d: all([
                    d.execute_script(
                        f"return document.getElementById('GLUXZipUpdateInput_0').value === '{zip_code[:3]}'"),
                    d.execute_script(
                        f"return document.getElementById('GLUXZipUpdateInput_1').value === '{zip_code[3:]}'")
                ]),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

            # Step 6: 检查是否有成功确认 DOM


            logging.info(f"邮编输入成功 | 耗时: {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")


            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "GLUXZipUpdateInput_0"))
                )

                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "GLUXZipUpdateInput_1"))
                )

                return self._input_ca_zip_code(zip_code, attempt + 1)

            raise

    def _input_uk_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_uk_zip_code(zip_code, attempt + 1)

            raise

    def _input_fr_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_fr_zip_code(zip_code, attempt + 1)

            raise

    def _input_de_zip_code(self, zip_code: str, attempt: int = 0):
        """增强版邮编输入（带前端事件触发）"""
        start_time = time.time()
        try:
            logging.info(f"⌨️ 开始输入邮编 {zip_code} (尝试#{attempt})")

            # 步骤1：定位输入框并等待交互就绪
            zip_input = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((self.locators.get('zip_input'))),
                "邮编输入框定位超时"
            )

            # 步骤2：三重清空策略
            for _ in range(3):
                zip_input.send_keys(Keys.CONTROL + 'a')
                zip_input.send_keys(Keys.DELETE)
                self.driver.execute_script("arguments[0].value = '';", zip_input)
                time.sleep(0.15)

            # 步骤3：模拟人工输入（带前端事件触发）
            self.driver.execute_script(f"""
                const input = document.getElementById('GLUXZipUpdateInput');
                input.value = '{zip_code}';

                // 触发完整的事件序列
                ['focus', 'keydown', 'keypress', 'input', 'keyup', 'change'].forEach(eventType => {{
                    input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                }});

                // 触发亚马逊的特定验证逻辑
                if (window.GLUX && GLUX.validatePostalCode) {{
                    GLUX.validatePostalCode(input.value);
                }}
            """)

            # 步骤4：验证输入结果（带重试机制）
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return document.getElementById('GLUXZipUpdateInput').value === arguments[0];
                """, str(zip_code)),
                f"邮编输入验证失败：预期 {zip_code}"
            )

            logging.info(f"邮编输入完成 | 耗时: {time.time() - start_time:.2f}s")

            # self.driver.execute_script("""
            #     document.querySelector('button[data-action=GLUXPostalUpdateAction]').focus();
            #     setTimeout(() => {
            #         document.activeElement.blur();
            #     }, 500);
            # """)
            # time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            logging.error(f"邮编输入失败: {str(e)}")
            # self._capture_debug_snapshot(f"zip_input_failure_attempt{attempt}")

            # 自动恢复机制
            if "stale element" in str(e).lower():
                self.driver.execute_script("location.reload(true);")
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((self.locators.get('zip_input')))
                )
                return self._input_de_zip_code(zip_code, attempt + 1)

            raise

    def _manual_debug_helper(self, element):
        """开启手动调试模式"""
        try:
            # 保存路径处理
            debug_dir = os.path.join(os.getcwd(), "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # 时间戳生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 截图保存
            screenshot_path = os.path.join(debug_dir, f"debug_{timestamp}.png")
            self.driver.save_screenshot(screenshot_path)

            # HTML保存
            html_path = os.path.join(debug_dir, f"debug_{timestamp}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)

            # 元素高亮
            original_style = self.driver.execute_script("""
                const oldStyle = arguments[0].getAttribute('style');
                arguments[0].style.border = '3px solid red';
                arguments[0].style.boxShadow = '0 0 20px rgba(255,0,0,0.5)';
                return oldStyle;
            """, element)

            # 打开浏览器（需要platform模块）
            current_url = self.driver.current_url
            system_name = platform.system()

            if system_name == 'Windows':
                os.startfile(current_url)  # Windows
            elif system_name == 'Darwin':
                subprocess.Popen(['open', current_url])  # macOS
            else:
                subprocess.Popen(['xdg-open', current_url])  # Linux

            # 交互提示
            logging.critical(" 进入手动调试模式 ")
            input("检查完成后按 Enter 继续...")

            # 恢复样式
            self.driver.execute_script("arguments[0].setAttribute('style', arguments[1])",
                                       element, original_style)

        except Exception as e:
            logging.error(f"手动调试助手异常: {str(e)}")
            raise

    def _action_chain_click(self, element):
        """增强版ActionChain点击，带异常捕获"""
        try:
            if not element.is_displayed() or not element.is_enabled():
                logging.warning("元素不可见或不可用，无法点击")
                raise Exception("元素不可见或不可用")

            logging.info("尝试ActionChain点击元素")
            ActionChains(self.driver) \
                .move_to_element(element) \
                .pause(0.3) \
                .click_and_hold() \
                .pause(0.2) \
                .release() \
                .perform()
        except Exception as e:
            logging.warning(f"ActionChain点击失败: {e}")
            raise

    def _is_success_text_present_us(self, zip_code):
        """快速检测成功文本"""
        try:
            apply = self.driver.find_element(
                By.XPATH,
                "//span[text()='Apply']/preceding-sibling::input[@type='submit']"

            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except:
            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.XPATH,
                                                   '//*[@id="GLUXHiddenSuccessSelectedAddressPlaceholder"]').text
                if zip_num == zip_code:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _is_success_text_present_es(self, zip_code):
        """快速检测成功文本"""
        try:
            apply = self.driver.find_element(
                By.XPATH,
                "//span[text()='Confirmar']/preceding-sibling::input[@type='submit']"
            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except Exception:
            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            hecho_button = WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Hecho']")))

            # 点击“Hecho”按钮关闭窗口
            hecho_button.click()

            # 等待提示文本出现，最长10秒
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            if flag.is_displayed():
                return True
            else:
                return False
        except Exception:
            try:
                zip_num = self.driver.find_element(By.XPATH, '//*[@id="glow-ingress-line2"]').text
                #print('zip_num:', zip_num)
                #print('zip_code:', zip_code)
                if zip_code in zip_num:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
                    return False
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False

    def _is_success_text_present_it(self, zip_code):
        """快速检测成功文本"""
        try:
            apply = self.driver.find_element(
                By.XPATH,
                "//span[@data-action='GLUXPostalUpdateAction']//input[@type='submit']"

            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except:
            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            hecho_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Fatto']")))

            # 点击“Hecho”按钮关闭窗口
            hecho_button.click()
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.XPATH,
                                                   '//*[@id="glow-ingress-line2"]').text
                print('zip_num', zip_num)
                print('zip_code',zip_code)
                if zip_code in zip_num:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _is_success_text_present_ca(self, zip_code):

        """快速检测邮编设置是否成功（适配加拿大站）"""
        try:
            # 如果设置按钮还可见，说明还没成功关闭
            apply_visible = self.driver.find_element(
                By.XPATH,
                "//input[@type='submit' and contains(@aria-labelledby, 'GLUXZipUpdate-announce')]"
            ).is_displayed()
            time.sleep(2)
            if apply_visible:
                logging.info("未成功点击，邮编未生效")
                return False
        except Exception:
            logging.info("Apply 按钮已消失，可能设置成功，继续验证")

        except:
            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.XPATH,
                                                   '//*[@id="GLUXHiddenSuccessSelectedAddressPlaceholder"]').text
                if zip_num == zip_code:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _is_success_text_present_uk(self, zip_code):
        """快速检测成功文本"""
        try:
            apply = self.driver.find_element(
                By.XPATH,
                "//span[@id='GLUXZipUpdate']//input[@type='submit' and contains(@class, 'a-button-input')]"
            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except:

            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.XPATH,
                                                   '//*[@id="glow-ingress-line2"]').text
                print(f'zip_num:{zip_num}')
                print(f'zip_code:{zip_code}')
                if zip_code[:4] in zip_num:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _is_success_text_present_fr(self, zip_code):
        """更鲁棒的成功文本检测"""
        try:
            apply = self.driver.find_element(
                By.XPATH, "//span[@id='GLUXZipUpdate-announce' and text()='Actualiser']/parent::span/input"
            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except:
            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.ID, "glow-ingress-line2").text.strip()
                print(f'zip_num:{zip_num}')
                print(f'zip_code:{zip_code}')
                if zip_code in zip_num:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _is_success_text_present_de(self, zip_code):
        """快速检测成功文本"""
        try:
            apply = self.driver.find_element(
                By.XPATH,
                "//input[@type='submit' and @aria-labelledby='GLUXZipUpdate-announce']"
            ).is_displayed()
            if apply:
                logging.info(f"未成功点击，邮编未生效")
                return False
        except:

            logging.info(f"点击设置成功")
            logging.info(f"验证邮编状态")

        try:
            flag = self.driver.find_element(
                By.XPATH,
                "//div[contains(text(), '我们将使用您选择的位置显示在')  or contains(text(),'We will use your selected location to show all')]"
            )
            return flag
        except:
            try:
                zip_num = self.driver.find_element(By.ID, "GLUXHiddenSuccessSelectedAddressPlaceholder").text
                if zip_code in zip_num:
                    logging.info('点击邮编设置后生效')
                    return True
                else:
                    logging.info('点击设置后邮编未生效')
            except Exception as e:
                logging.error(f"点击设置邮编后检测文本失败: {str(e)}")
                return False
            return False

    def _check_success_state(self, target_zip, wait_time=3):
        """优化后的快速状态检测"""
        try:
            if self.nation == 'us':
                # 优先检测文本内容
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_us(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'ca':
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_ca(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'uk':
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_uk(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'fr':
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_fr(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'de':
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_de(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'it':
                # 优先检测文本内容
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_it(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            elif self.nation == 'es':
                # 优先检测文本内容
                if WebDriverWait(self.driver, wait_time).until(
                        lambda d: self._is_success_text_present_es(target_zip)
                ):
                    logging.info(" 成功文本内容已显示")
                    return True
            # 二次验证邮编匹配
            actual_zip = self.driver.find_element(
                By.ID, "GLUXHiddenSuccessSelectedAddressPlaceholder"
            ).text
            if actual_zip == target_zip:
                logging.info(f"邮编匹配成功: {target_zip}")
                return True

            return False
        except Exception as e:
            logging.warning(f"快速验证失败: {str(e)}")
            return False

    def _highlight_element(self, element, color="red"):
        """元素高亮辅助"""
        self.driver.execute_script(
            f"arguments[0].style.outline='2px solid {color}';"
            f"arguments[0].style.boxShadow='0 0 10px {color}';",
            element
        )

    def _retry_click_with_validation_us(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_us(target_zip):
                        logging.info("检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f"[{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_es(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_es(target_zip):
                        logging.info("检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f"[{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_it(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_it(target_zip):
                        logging.info("检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f"[{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_ca(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_ca(target_zip):
                        logging.info("检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f"[{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_uk(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_uk(target_zip):
                        logging.info(" 检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f" [{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_fr(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f"[{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_fr(target_zip):
                        logging.info("检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f"[{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f"⚠️ [{strategy_name}] 已达最大重试次数")

        return False

    def _retry_click_with_validation_de(self, element, target_zip, retries=2):
        """优化后的点击验证流程"""
        # 调整点击策略顺序，优先ActionChain
        click_strategies = [
            ("ActionChain点击", lambda: self._action_chain_click(element)),
            ("标准点击", lambda: element.click()),
            ("JS点击", lambda: self.driver.execute_script("arguments[0].click();", element)),
            ("物理点击", lambda: element.send_keys("\n"))
        ]

        for strategy_name, strategy in click_strategies:
            for retry in range(retries):
                try:
                    logging.info(f" 尝试 [{strategy_name}] (第{retry + 1}次)")
                    strategy()
                    time.sleep(1)

                    # 快速验证（缩短等待时间）
                    if self._check_success_state(target_zip, wait_time=5):  # 从5秒缩短到3秒
                        logging.info(f" [{strategy_name}] 点击成功并通过验证")
                        return True

                    # 如果已出现成功文本但未完全加载，提前返回
                    if self._is_success_text_present_de(target_zip):
                        logging.info(" 检测到成功文本，提前结束尝试")
                        return True

                except Exception as e:
                    logging.warning(f" [{strategy_name}] 失败: {str(e)}")
                    self._highlight_element(element, color="orange")

            logging.warning(f" [{strategy_name}] 已达最大重试次数")

        return False

    def _check_zip_display(self, target_zip):
        """检查页面显示的邮编"""
        try:
            location_element = WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.ID, "glow-ingress-line2"))
            )
            displayed_zip = re.search(r"\b\d{5}\b", location_element.text).group()
            logging.info(f"页面显示邮编: {displayed_zip} | 预期: {target_zip}")
            return displayed_zip == target_zip
        except:
            return False

    def _verify_zip_change(self, target_zip):
        """综合验证邮编修改结果（简化版）"""
        # 先刷新页面确保状态更新
        self.driver.refresh()

        try:
            # 单一验证条件
            verification_passed = self._check_zip_display(target_zip)

            logging.info(f"验证结果: {'成功' if verification_passed else '失败'}")
            return verification_passed

        except Exception as e:
            logging.error(f"验证过程中出现异常: {str(e)}")
            return False

    def close(self):
        """关闭浏览器并执行深度清理"""
        try:
            if self.driver:
                # 1. 清理浏览器数据
                try:
                    self.driver.execute_cdp_cmd('Network.clearBrowserCache', {})
                    self.driver.execute_cdp_cmd('Storage.clearDataForOrigin', {
                        "origin": '*',
                        "storageTypes": 'all'
                    })
                    logging.info("CDP清理成功")
                except Exception as e:
                    logging.warning(f"CDP清理失败: {str(e)}")

                # 2. 清理本地存储
                self.driver.execute_script("""
                    try {
                        localStorage.clear();
                        sessionStorage.clear();
                        indexedDB.deleteDatabase('localforage');
                    } catch(e){}
                """)
                logging.info("本地存储清理成功")

                # 3. 清理cookies
                self.driver.delete_all_cookies()
                logging.info("cookies清理成功")

                # 4. 关闭浏览器
                self.driver.close()
                self.driver.quit()
                logging.info("浏览器会话关闭成功")

                # 5. 强制清理残留进程（增强版）
                cleanup_commands = {
                    'Windows': [
                        'taskkill /f /im chromedriver.exe',
                        'taskkill /f /im chrome.exe'
                    ],
                    'Darwin': [
                        'pkill -f chromedriver',
                        'pkill -f Chrome'
                    ],
                    'Linux': [
                        'pkill -f chromedriver',
                        'pkill -f chrome'
                    ]
                }
                if cleanup_commands.get(platform.system(), []) == []:
                    logging.info("无残留进程")
                else:
                    for cmd in cleanup_commands.get(platform.system(), []):
                        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
                    logging.info("清理残留进程完成")

                # 6. 清理临时文件
                temp_files = [
                    'current_captcha.png',
                    'amazon_helper.log'
                ]
                if temp_files == []:
                    logging.info("无临时文件")
                else:
                    for file in temp_files:
                        try:
                            os.remove(file)
                        except:
                            logging.warning(f"无法删除临时文件: {file}")
                    logging.info("清理文件完成")


        except Exception as e:
            logging.error(f"关闭清理异常: {str(e)}")

    def _get_random_zip(self):
        """获取随机邮编并记录"""
        new_zip = random.choice(self.zip_list)
        logging.info(f"生成随机邮编: {new_zip}")
        return new_zip


    def _detect_throttling(self, driver):
        """检测请求限制提示（支持多语言版本）"""
        throttling_indicators = [
            # 英文提示（不区分大小写）
            (By.XPATH,
             "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'request was throttled')]"),
            # 另一种英文提示
            (By.XPATH,
             "//h1[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'please try again later')]"),
            # CSS选择器定位警告容器
            (By.CSS_SELECTOR, "div.a-box-inner.a-alert-container"),
            (By.XPATH, "//span[contains(text(),'请求过于频繁')]"),  # 中文提示
            (By.XPATH, "//span[contains(text(),'demasiadas solicitudes')]")  # 西班牙语提示
        ]

        for locator in throttling_indicators:
            try:
                element = driver.find_element(*locator)
                if element.is_displayed():
                    # 记录详细节流信息
                    logging.error(f"节流检测命中 | 内容: {element.text[:50]}...")
                    return True
            except (NoSuchElementException, StaleElementReferenceException):
                continue
        return False

    def _recover_from_throttling(self):
        '''
        完全关闭当前浏览器实例
        重新初始化新的浏览器实例
        '''
        logging.critical("进入节流恢复模式")
        recovery_steps = [
            self._clear_tracking_elements,  # 清理跟踪元素
            self._rotate_user_agent,  # 轮换用户代理
            # 随机等待
            lambda: self.driver.delete_all_cookies(),  # 清理Cookies
            lambda: self.driver.execute_script("window.localStorage.clear();")  # 清理本地存储
        ]
        [step() for step in recovery_steps]  # 使用列表推导式顺序执行所有恢复操作浏览器重置
        self.driver.close()
        self.driver.quit()
        self.driver = self._init_driver()

    def _clear_tracking_elements(self):
        """清理可能用于跟踪的元素"""
        self.driver.execute_script("""
               // 删除亚马逊跟踪像素
               document.querySelectorAll('img[src*="fls-na.amazon.com"]').forEach(img => img.remove());
               // 清除指纹存储
               localStorage.removeItem('adobe-id');
               sessionStorage.removeItem('session-token');
           """)

    def _rotate_user_agent(self):
        """轮换User-Agent防检测"""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        ]
        new_agent = random.choice(agents)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": new_agent})


if __name__ == '__main__':
    data = pd.read_csv('/asin.csv')
    asinlist = data['asin'].values.tolist()
    logging.basicConfig(
            level=logging.INFO, # 设置日志级别
             format = '%(asctime)s-%(name)s-%(levelname)s -%(message)s',  # 设置输出格式
            datefmt = '%Y-%m-%d %H:%M:%S',  # 设置时间格式
            encoding = 'utf-8'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    proxy_manager = ProxyManager()
    amazon_helper = AmazonScraper(proxy_manager, nation='CA')

    db_config = {

    "host": "",
    "user": "",
    'password': "",
    'database': ""
    }
    ProductDetailScraper = ProductDetailScraper(amazon_helper, db_config, nation = "CA")

    for i in range(10):

        ProductDetailScraper.scrape_and_save(asinlist[i])
