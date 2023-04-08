import warnings
from typing import Optional

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from webdriver_manager.chrome import ChromeDriverManager


class DriverFactory:
    @classmethod
    def get_driver(
        cls, path: Optional[str] = None, headless: bool = True, verbose: bool = True
    ) -> WebDriver:
        """
        Initialize selenium.webdriver.Chrome object and checks
        its compatibility with chrome version
        """
        opts = cls._get_options(headless=headless)
        driver = cls._get_driver(opts=opts, driver_path=path, verbose=verbose)
        cls._check_compatibility(driver=driver, verbose=verbose)
        return driver

    @classmethod
    def _check_compatibility(cls, driver: WebDriver, verbose: bool = True) -> None:
        """
        Checks driver's version compatibility with browser version,
        if version is different, raises a warning

        Parameters:
            verbose : bool, deafult = True
                If true prints versions details
        """
        # check compatibility with chrome browser
        details = cls._get_details(driver)
        browser_version = details["browser_version"].split(".")[0]
        driver_version = details["driver_version"].split(".")[0]
        if browser_version != driver_version:
            warnings.warn(
                """Browser's version might be incompatible with
                        driver's version""".replace(
                    "  ", ""
                )
            )
        if verbose:
            for key, value in details.items():
                print(f"{key}: {value}")

    @staticmethod
    def _get_details(driver: WebDriver) -> dict:
        """
        Uses driver's capability attribute to extract information about
        driver and browser
        """
        browser_name = driver.capabilities["browserName"]
        browser_version = driver.capabilities["browserVersion"]
        driver_version = driver.capabilities[browser_name][
            f"{browser_name}driverVersion"
        ].split(" ")[0]
        details = {"driver_version": driver_version, "browser_version": browser_version}
        return details

    @staticmethod
    def _get_options(headless: bool = True) -> Options:
        """
        Returns selenium.webdriver.chrome.options.Options for driver

        Parameters:
            headless : bool, deafult = True
                If false driver runs browser in the background
        """
        opts = Options()
        opts.headless = headless
        opts.add_experimental_option(
            "excludeSwitches", ["enable-logging", '"disable-popup-blocking"']
        )
        return opts

    @staticmethod
    def _get_driver(
        opts: Options, driver_path: Optional[str] = None, verbose: bool = True
    ) -> WebDriver:
        """
        Returns selenium.webdriver.Chrome instance,
        if path to driver exe is not specified, downloads the lastest version
        """
        if driver_path is None:
            # download latest driver
            driver_path = ChromeDriverManager().install()

        if verbose:
            print(f"driver: {driver_path}")

        return Chrome(driver_path, options=opts)
