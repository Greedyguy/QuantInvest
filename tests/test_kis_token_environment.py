import importlib
import importlib.util
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


TOKEN_MANAGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "kiwoom_api"
    / "core"
    / "kis_token_manager.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kis_token_manager_under_test",
    TOKEN_MANAGER_PATH,
)
token_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token_module)
KISTokenManager = token_module.KISTokenManager
get_token_manager = token_module.get_token_manager


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload
        self.text = str(payload)

    def json(self):
        return self.payload


class KISTokenEnvironmentTest(unittest.TestCase):
    def setUp(self):
        token_module._token_managers.clear()

    def tearDown(self):
        token_module._token_managers.clear()

    def test_explicit_real_environment_overrides_missing_env_setting(self):
        with patch.dict(os.environ, {}, clear=True), TemporaryDirectory() as tmp:
            manager = KISTokenManager(
                appkey="real-key",
                appsecret="secret",
                virtual_account=False,
                cache_dir=Path(tmp),
            )

        self.assertFalse(manager.virtual_account)
        self.assertEqual(
            manager.base_url,
            "https://openapi.koreainvestment.com:9443",
        )
        self.assertIn("shared_token_real_", manager.cache_file.name)

    def test_real_and_virtual_environments_never_share_manager(self):
        real_manager = get_token_manager(
            appkey="same-key",
            appsecret="secret",
            virtual_account=False,
        )
        virtual_manager = get_token_manager(
            appkey="same-key",
            appsecret="secret",
            virtual_account=True,
        )

        self.assertIsNot(real_manager, virtual_manager)
        self.assertFalse(real_manager.virtual_account)
        self.assertTrue(virtual_manager.virtual_account)
        self.assertNotEqual(real_manager.cache_file, virtual_manager.cache_file)

    def test_real_connector_selects_real_token_server_without_env_flag(self):
        environment = {
            "KIS_APP_KEY": "real-key",
            "KIS_APP_SECRET": "secret",
            "KIS_ACCOUNT": "12345678",
        }
        with patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()):
            connector_module = importlib.import_module(
                "kiwoom_api.core.korea_investment_connector"
            )
            connector = connector_module.KoreaInvestmentConnector(
                virtual_account=False
            )

        self.assertFalse(connector.token_manager.virtual_account)
        self.assertEqual(connector.token_manager.base_url, connector.BASE_URL)
        self.assertEqual(
            connector.BASE_URL,
            "https://openapi.koreainvestment.com:9443",
        )

    def test_cached_token_from_other_environment_is_rejected(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            manager = KISTokenManager(
                appkey="same-key",
                appsecret="secret",
                virtual_account=False,
                cache_dir=cache_dir,
            )
            manager.cache_file.write_text(
                '{"access_token":"wrong","expires_at":9999999999,'
                '"environment":"virtual","appkey_fingerprint":"'
                + manager.appkey_fingerprint
                + '"}',
                encoding="utf-8",
            )

            self.assertFalse(manager._load_cached_token())
            self.assertIsNone(manager.access_token)

    def test_minute_rate_limit_waits_and_retries_once(self):
        rate_limited = FakeResponse(
            403,
            {
                "error_code": "EGW00133",
                "error_description": "접근토큰 발급 잠시 후 다시 시도하세요(1분당 1회)",
            },
        )
        success = FakeResponse(
            200,
            {"access_token": "new-token", "expires_in": 86400},
        )

        with TemporaryDirectory() as tmp:
            manager = KISTokenManager(
                appkey="real-key",
                appsecret="secret",
                virtual_account=False,
                cache_dir=Path(tmp),
            )
            with (
                patch("requests.post", side_effect=[rate_limited, success]) as post,
                patch.object(token_module.time, "sleep") as sleep,
            ):
                token = manager._request_new_token()

        self.assertEqual(token, "new-token")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(65)


if __name__ == "__main__":
    unittest.main()
