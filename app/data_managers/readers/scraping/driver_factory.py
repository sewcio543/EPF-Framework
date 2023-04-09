import warnings
from typing import Optional

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from webdriver_manager.chrome import ChromeDriverManager

DRIVER_VERSION = "driver_version"
BROWSER_VERSION = "browser_version"


class DriverFactory:
    def __init__(
        self, path: Optional[str] = None, headless: bool = True, verbose: bool = True
    ) -> None:
        """Initializes DriverFactory

        Args:
            path: str
                Executable path for Webdriver. Defaults to None.
            headless: bool, optional
                Whether driver should run in headless mode. Defaults to True.
                If headless is False, browser will be open.
            verbose: bool, optional
                If verbose is set to True, additional information will be printed out.
                Defaults to True.
        """
        self.path = path
        self.headless = headless
        self.verbose = verbose

    def get(self) -> WebDriver:
        """
        Initialize selenium.webdriver.Chrome object and checks
        its compatibility with chrome version

        Returns:
            Webdriver: selenium.webdriver.Chrome
        """
        opts = self._get_options()
        driver = self._get_driver(options=opts)
        self._check_compatibility(driver=driver)
        return driver

    def _check_compatibility(self, driver: WebDriver) -> None:
        """
        Checks driver's version compatibility with browser version,
        if version is different, raises a warning

        Parameters:
            driver : Chrome
                Selenium Chrome webdriver object
        """
        # check compatibility with chrome browser
        details = self._get_details(driver)
        browser_version = details[BROWSER_VERSION].split(".")[0]
        driver_version = details[DRIVER_VERSION].split(".")[0]
        if browser_version != driver_version:
            warnings.warn(
                "Browser's version might be incompatible with driver's version"
            )
        if self.verbose:
            for key, value in details.items():
                print(f"{key}: {value}")

    def _get_details(self, driver: WebDriver) -> dict:
        """
        Uses driver's capability attribute to extract information about
        driver and browser

        Parameters:
            driver : Chrome
                Selenium Chrome webdriver object

        Returns:
            dict: driver and browser version information
        """
        browser_name = driver.capabilities["browserName"]
        browser_version = driver.capabilities["browserVersion"]
        driver_version = driver.capabilities[browser_name][
            f"{browser_name}driverVersion"
        ].split(" ")[0]
        details = {DRIVER_VERSION: driver_version, BROWSER_VERSION: browser_version}
        return details

    def _get_options(self) -> Options:
        """
        Gets driver Options

        Parameters:
            headless : bool, deafult = True
                If false driver runs browser in the background

        Returns:
            Options: Selenium Webdriver Options
        """
        opts = Options()
        opts.headless = self.headless
        opts.add_experimental_option(
            "excludeSwitches", ["enable-logging", "disable-popup-blocking"]
        )
        return opts

    def _get_driver(self, options: Options) -> WebDriver:
        """
        Returns selenium.webdriver.Chrome instance,
        if path to driver exe is not specified, downloads the lastest version

        Parameters:
            options: Options
                Selenium Webdriver Options

        Returns:
            Webdriver: selenium.webdriver.Chrome
        """
        if self.path is None:
            # download latest driver
            self.path = ChromeDriverManager().install()

        if self.verbose:
            print(f"driver: {self.path}")

        return Chrome(executable_path=self.path, options=options)
