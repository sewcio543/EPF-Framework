from argparse import Namespace

scraper_ns = Namespace(
    CHROME_DRIVER='chromedriver.exe',
    DEFAULT_PARSER='lxml',
    PRIVACY_XPATH='//*[@id="onetrust-accept-btn-handler"]'
)
