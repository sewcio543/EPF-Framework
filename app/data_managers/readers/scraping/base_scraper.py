import time
from typing import Optional, Union

from bs4.element import Tag
from selenium.common.exceptions import (
    InvalidArgumentException,
    InvalidSelectorException,
    NoSuchElementException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from ...namespaces import scraper_ns
from .driver_factory import DriverFactory
from .tag_extractor import TagExtractor


class BaseScraper(TagExtractor):
    def __init__(
        self,
        driver: Optional[str] = None,
        headless: bool = True,
        wait_time: int = 5,
        verbose: bool = True,
        load_time: float = 0.1,
    ) -> None:
        """
        Initialize selenium.webdriver.Chrome object

        Parameters:
            driver : str, optional
                Path to .exe file with driver for Google Chrome browser
            headless : bool, default True
                If False driver runs in visible mode
            wait_time: int, default 0
                Seconds to wait for element fo be visible
            verbose: bool, default True
                If True additional printouts will be added
            load_time: float, default 0.1
                Seconds to wait after directing to page
        """
        self._load_sleep = load_time
        self._wait_time = wait_time
        self._factory = DriverFactory(path=driver, headless=headless, verbose=verbose)
        self._driver = self._factory.get()

    @property
    def current_url(self) -> str:
        return self._driver.current_url

    def get_source(
        self,
        link: str,
        parser: str = scraper_ns.DEFAULT_PARSER,
        end_session: bool = True,
    ) -> Tag:
        """
        Execute driver.get method, loads a webpage in a current
        browser session and returns page markup

        Parameters:
            link : str
                link to webpage
            parser: str, optional
                BeautifulSoup parser, default lxml
            end_session: bool, deafult = True
                If false browser session is not closed
        """
        try:
            self._driver.get(link)
        except InvalidArgumentException:
            print("GET REQUEST FAILED - INVALID PAGE")
            self.quit()
        except AttributeError:
            print("AttributeError - probably tag was not found")
            self.quit()

        self._accept_privacy()
        # wait for page to load
        time.sleep(self._load_sleep)
        parsed_markup = self._get_markup(parser=parser)
        if end_session:
            self.quit()
        return parsed_markup

    def _get_markup(
        self,
        element: Optional[WebElement] = None,
        parser: str = scraper_ns.DEFAULT_PARSER,
    ) -> Tag:
        """
        Returns BeautifulSoup markup from WebElement if specified, otherwise
        from body tag.
        """
        if element is None:
            raw_html = self._driver_get_body()
        else:
            raw_html = element.get_attribute("innerHTML")
        parsed_markup = self._to_bs(raw_html, parser)
        return parsed_markup

    def _driver_get_body(self) -> str:
        """Returns raw body tag from curent session"""
        assert self._driver.session_id, "Session doesn't exist"
        body = self._get_element(selector="body")
        assert body is not None, "body selector not found"
        raw_html = body.get_attribute("innerHTML")
        return raw_html

    def _accept_privacy(self) -> None:
        """Accepts Chrome privacy popup if present"""
        privacy_el = self._get_element(
            selector=scraper_ns.PRIVACY_XPATH, wait=False, by=By.XPATH
        )
        if privacy_el:
            self._click_element(privacy_el)

    def _get_elements(
        self,
        selector: Union[str, dict],
        wait: bool = True,
        by: str = By.CSS_SELECTOR,
    ) -> list[WebElement]:
        """
        Returns list of WebElements matching the selector

        Parameters:
            selector : str
                selector of the tag
            wait: bool, default True
                whether to wait for element to be visible
            by: str, default By.CSS_SELECTOR
                Type of selector

        Returns:
            WebElement: All found WebElements matching the selector
        """
        if isinstance(selector, dict):
            selector = self._tag_to_selector(tag_dict=selector)
        if wait:
            try:
                # TODO: check for speed
                elements = WebDriverWait(
                    driver=self._driver, timeout=self._wait_time
                ).until(EC.visibility_of_any_elements_located((by, selector)))

            except InvalidSelectorException as e:
                print("INVALID CSS SELECTOR")
                raise InvalidSelectorException from e

            except NoSuchElementException as e:
                print("ELEMENT NOT FOUND")
                raise NoSuchElementException from e

            except TimeoutException as e:
                raise TimeoutException from e
        else:
            elements = self._driver.find_elements(by, selector)

        return elements

    def _get_element(
        self,
        selector: str,
        wait: bool = True,
        by: str = By.CSS_SELECTOR,
    ) -> WebElement | None:
        """
        Returns first WebElement matching the selector

        Parameters:
            selector : str
                selector of the tag
            wait: bool, default True
                whether to wait for element to be visible
            by: str, default By.CSS_SELECTOR
                Type of selector

        Returns:
            WebElement: First WebElement matching the selector
        """
        elements = self._get_elements(selector=selector, wait=wait, by=by)
        return elements[0] if elements else None

    def _switch_to_new_window(self):
        """Switches driver to the last open window"""
        new_window = self._driver.window_handles[-1]
        self._driver.switch_to.window(new_window)

    def _close_window(self) -> None:
        """Closes current window or tab switches back to last open one"""
        # closes current window or tab
        self.close()
        # goes back to open page
        self._switch_to_new_window()

    def _click_element(self, element: WebElement):
        """Clicks WebElement object"""
        self._driver.execute_script("arguments[0].click();", element)

    def quit(self) -> None:
        """Quits driver session using driver.quit method"""
        self._driver.quit()

    def close(self) -> None:
        """Closes current page using driver.close method"""
        self._driver.close()
