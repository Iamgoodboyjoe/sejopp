import getpass
import json
import logging
import os
import pickle
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
import tqdm

try:
    from IPython.display import display, HTML, Javascript
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

from helpful_functions import (
    expand_dict_columns,
    save_pnl,
    save_simulation_result,
    save_yearly_stats,
)

DEV = False

class SingleSession(requests.Session):
    _instance = None
    _lock = threading.Lock()
    _relogin_lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not self._initialized:
            super(SingleSession, self).__init__(*args, **kwargs)
            self._initialized = True

    def get_relogin_lock(self):
        return self._relogin_lock

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ace")
    if logger.hasHandlers():
        return logger
    level = logging.DEBUG if DEV else logging.INFO
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter("\n[%(asctime)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if DEV:
        file_handler = logging.FileHandler("log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logger()

DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")

def get_credentials_from_notebook() -> tuple[str, str]:
    if not JUPYTER_AVAILABLE:
        return None
    try:
        html_header = """
        <style>
            .credential_form_container {
                max-width: 400px;
                font-family: Arial, sans-serif;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            .credential_form_title {
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                color: #333;
            }
            .credential_form_desc {
                font-size: 14px;
                color: #666;
                margin-bottom: 15px;
            }
            .credential_form_note {
                font-size: 13px;
                color: #28a745;
                background: #d4edda;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        </style>
        
        <div class="credential_form_container">
            <div class="credential_form_title">[LOCK] BRAIN Platform Credentials</div>
            <div class="credential_form_desc">Enter your credentials below when prompted</div>
            <div class="credential_form_note">[OK] Credentials are validated and securely stored locally</div>
        </div>
        """
        display(HTML(html_header))
        print("\n" + "="*60)
        sys.stdout.flush()
        email = input("Email: ")
        sys.stdout.flush()
        password = getpass.getpass(prompt="Password: ")
        sys.stdout.flush()
        print("="*60 + "\n")
        email = email.strip()
        if not email or not password:
            print("Email and password cannot be empty")
            return None
        if "@" not in email:
            print("Please enter a valid email address")
            return None
        print("Credentials received successfully!")
        return (email, password)
    except Exception as e:
        logger.warning(f"HTML form display failed ({e}), falling back to terminal input")
        return None

def get_credentials(force_new: bool = False) -> tuple[str, str]:
    credential_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL")
    credential_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD")
    credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
    credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")

    if not force_new and Path(credentials_file_path).exists() and os.path.getsize(credentials_file_path) > 2:
        with open(credentials_file_path) as file:
            data = json.loads(file.read())
    else:
        os.makedirs(credentials_folder_path, exist_ok=True)
        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            if JUPYTER_AVAILABLE:
                try:
                    result = get_credentials_from_notebook()
                    if result is None:
                        raise Exception("HTML form cancelled or timed out")
                    email, password = result
                except Exception as e:
                    logger.warning(f"HTML form failed ({e}), falling back to terminal input")
                    print("\n" + "="*60)
                    print("ENTER YOUR CREDENTIALS")
                    print("="*60)
                    sys.stdout.flush()
                    email = input("\nEmail: ")
                    sys.stdout.flush()
                    password = getpass.getpass(prompt="Password: ")
                    sys.stdout.flush()
                    print("="*60 + "\n")
            else:
                print("\n" + "="*60)
                print("ENTER YOUR CREDENTIALS")
                print("="*60)
                sys.stdout.flush()
                email = input("\nEmail: ")
                sys.stdout.flush()
                password = getpass.getpass(prompt="Password: ")
                sys.stdout.flush()
                print("="*60 + "\n")
        data = {"email": email, "password": password}
        with open(credentials_file_path, "w") as file:
            json.dump(data, file)
    return (data["email"], data["password"])

SESSION_CACHE_PATH = os.path.join(os.path.expanduser("~"), "secrets", "shared_brain_session.pkl")

def save_session_to_disk(session: SingleSession, remaining_seconds: int):
    try:
        os.makedirs(os.path.dirname(SESSION_CACHE_PATH), exist_ok=True)
        expiry_timestamp = time.time() + remaining_seconds
        payload = {
            "cookies": session.cookies.get_dict(),
            "expiry": expiry_timestamp
        }
        temp_path = SESSION_CACHE_PATH + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, SESSION_CACHE_PATH)
        session.local_expiry = expiry_timestamp
        logger.debug(f"Session persisted. Expiry: {datetime.fromtimestamp(expiry_timestamp).strftime('%H:%M:%S')}")
    except Exception as e:
        logger.warning(f"Failed to save session to disk: {e}")
        
def load_session_from_disk(session: SingleSession) -> bool:
    if not os.path.exists(SESSION_CACHE_PATH):
        logger.debug("No session cache found.")
        return False
    file_size = os.path.getsize(SESSION_CACHE_PATH)
    if file_size == 0:
        logger.warning("Session cache is corrupted (0 bytes). Will re-authenticate.")
        return False
    try:
        with open(SESSION_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "cookies" in data:
            session.cookies.update(data["cookies"])
            session.local_expiry = data.get("expiry", 0)
            logger.debug(f"Loaded session from cache (expires in {data.get('expiry', 0) - time.time():.0f}s)")
            return True
        elif isinstance(data, dict):
            session.cookies.update(data)
            session.local_expiry = 0
            logger.debug("Loaded legacy session format (no expiry).")
            return True
        logger.warning(f"Session cache has unexpected format: {type(data)}")
        return False
    except (pickle.UnpicklingError, EOFError) as e:
        logger.warning(f"Session cache corrupted (pickle error: {e}). Re-authenticating...")
        try:
            os.remove(SESSION_CACHE_PATH)
        except:
            pass
        return False
    except Exception as e:
        logger.warning(f"Unexpected error loading session: {e}")
        return False

def start_session(force_refresh: bool = False) -> SingleSession:
    s = SingleSession()
    if not force_refresh and load_session_from_disk(s):
        remaining = check_session_timeout(s)
        if remaining > 60:
            expiry_str = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
            logger.info(f"Brain Session Active Expires: {expiry_str}\n---- Happy Quanting ----")
            return s
    s.auth = get_credentials(force_new=force_refresh)
    r = s.post(brain_api_url + "/authentication")
    if r.status_code == 401 and r.headers.get("WWW-Authenticate") == "persona":
        persona_url = urljoin(r.url, r.headers["Location"])
        print("\n" + "-+"*30)
        print("BIOMETRICS AUTHENTICATION REQUIRED")
        print("-+"*30)
        print(f"\n[LINK] Complete scan at this URL:\n{persona_url}\n")
        sys.stdout.flush()
        input(">>> Press ENTER after completing the scan...")
        count = 0
        max_retries = 3
        try:
            handshake = s.post(persona_url, timeout=15)
        except Exception as e:
            print(f"[WARNING] Connection error during handshake: {e}")
            return start_session(force_refresh=force_refresh)
        while True:
            try:
                if s.post(persona_url, timeout=15).status_code == 201:
                    try:
                        auth_res = s.get(brain_api_url + "/authentication", timeout=10).json()
                        server_seconds = int(auth_res["token"]["expiry"])
                        save_session_to_disk(s, server_seconds)
                    except Exception as e:
                        logger.warning(f"Could not verify fresh expiry: {e}. Defaulting to 4h.")
                        save_session_to_disk(s, 14400)
                    print("[OK] Biometric scan verified! Session authenticated.")
                    print("-"*60 + "\n")
                    sys.stdout.flush()
                    break
                else:
                    count += 1
                    if count >= max_retries:
                        print(f"[ERROR] Biometric authentication failed after {max_retries} attempts.")
                        print("-"*60 + "\n")
                        sys.stdout.flush()
                        return start_session(force_refresh=True)
                    print(f"[WAIT] Scan not completed yet. Please try again. (Attempt {count}/{max_retries})")
                    sys.stdout.flush()
                    input(">>> Press ENTER after completing the scan...")
            except Exception as e:
                count += 1
                if count >= max_retries:
                    print(f"[ERROR] Connection error after {max_retries} attempts: {e}")
                    print("-"*60 + "\n")
                    sys.stdout.flush()
                    return start_session(force_refresh=True)
                print(f"[WARNING] Connection error: {e}. Retrying... (Attempt {count}/{max_retries})")
                sys.stdout.flush()
                input(">>> Press ENTER to retry...")
    elif r.status_code not in [200, 201]:
        logger.error("\nIncorrect email or password. Please try again with correct credentials.\n")
        credentials_file_path = os.path.join(os.path.expanduser("~"), "secrets", "platform-brain.json")
        try:
            os.remove(credentials_file_path)
        except:
            pass
        return start_session(force_refresh=True)
    return s

def check_session_and_relogin(s):
    remaining = check_session_timeout(s)
    if remaining >= 600:
        return s
    load_session_from_disk(s)
    remaining_after_reload = check_session_timeout(s)
    if remaining_after_reload <= 0:
        now = time.time()
        last_alert = getattr(s, '_last_alert_time', 0)
        if (now - last_alert) > 600:
            logger.info("Session expired. Alerting user...")
            s._last_alert_time = now
        return s
    return s

def check_session_timeout(s: SingleSession) -> int:
    local_expiry = getattr(s, 'local_expiry', 0)
    if local_expiry == 0:
        load_session_from_disk(s)
        local_expiry = getattr(s, 'local_expiry', 0)
    now = time.time()
    local_remaining = int(local_expiry - now)
    if local_remaining > 300:
        return local_remaining
    try:
        response = s.get(brain_api_url + "/authentication", timeout=5)
        if response.status_code == 200:
            data = response.json()
            server_remaining = int(data.get("token", {}).get("expiry", 0))
            if server_remaining > 60:
                save_session_to_disk(s, server_remaining)
                return server_remaining
            return 0
        elif response.status_code == 401:
            return 0
    except:
        return max(0, local_remaining)
    return 0

class SessionExpiredError(Exception):
    pass

def verify_session_truth_silent(s: SingleSession) -> bool:
    try:
        response = s.get(f"{brain_api_url}/authentication", timeout=10)
        if response.status_code == 200:
            server_remaining = int(response.json()["token"]["expiry"])
            if server_remaining > 60:
                save_session_to_disk(s, server_remaining)
                return True
        return False
    except:
        return False

def get_simulation_result_json(s: SingleSession, alpha_id: str) -> dict:
    if alpha_id is None:
        return {}
    while True:
        response = s.get(f"{brain_api_url}/alphas/{alpha_id}", timeout=15)
        if response.status_code == 401:
            if verify_session_truth_silent(s):
                continue
            return {}
        if "retry-after" in response.headers:
            time.sleep(float(response.headers["Retry-After"]))
        else:
            try:
                return response.json()
            except Exception:
                return {}

def simulation_progress(s: SingleSession, simulate_response: requests.Response) -> dict:
    if simulate_response.status_code // 100 != 2:
        logger.error(
            "Simulation request failed (HTTP %s): %s",
            simulate_response.status_code,
            getattr(simulate_response, "text", "")[:1000],
        )
        return {"completed": False, "result": {}}
    url = simulate_response.headers.get("Location")
    if not url:
        logger.error("Simulation response did not include a Location header.")
        return {"completed": False, "result": {}}
    while True:
        if check_session_timeout(s) <= 0:
            if not verify_session_truth_silent(s):
                raise SessionExpiredError("Session expired during polling.")
        response = s.get(url, timeout=15)
        if response.status_code == 401:
            if verify_session_truth_silent(s):
                continue
            raise SessionExpiredError("Session expired during polling.")
        if response.status_code // 100 != 2:
            time.sleep(10)
            continue
        data = response.json()
        if data.get("status") in ["COMPLETE", "ERROR"]:
            break
        wait = float(response.headers.get("Retry-After", 10))
        time.sleep(wait)
    alpha_value = data.get("alpha")
    alpha_id = (
        alpha_value.get("id")
        if isinstance(alpha_value, dict)
        else alpha_value
    )
    if not alpha_id or data.get("status") == "ERROR":
        logger.error("Simulation finished without an alpha ID: %s", data)
        return {"completed": False, "result": {}}
    result = get_simulation_result_json(s, alpha_id)
    return {"completed": True, "result": result}

def multisimulation_progress(s: SingleSession, simulate_response: requests.Response) -> dict:
    if simulate_response.status_code // 100 != 2:
        return {"completed": False, "result": []}
    url = simulate_response.headers.get("Location")
    if not url:
        logger.error("Multi-simulation response did not include a Location header.")
        return {"completed": False, "result": []}
    while True:
        if check_session_timeout(s) <= 0:
            if not verify_session_truth_silent(s):
                raise SessionExpiredError("Session expired during multi-polling.")
        response = s.get(url, timeout=15)
        if response.status_code == 401:
            if verify_session_truth_silent(s):
                continue
            raise SessionExpiredError("Session expired during multi-polling.")
        if response.status_code // 100 != 2:
            logger.warning(
                "Multi-simulation polling returned HTTP %s; retrying.",
                response.status_code,
            )
            time.sleep(float(response.headers.get("Retry-After", 10)))
            continue
        try:
            data = response.json()
        except Exception:
            logger.warning("Multi-simulation polling returned non-JSON data; retrying.")
            time.sleep(float(response.headers.get("Retry-After", 10)))
            continue
        if data.get("status") in ["COMPLETE", "WARNING", "ERROR", "FAILED", "CANCELLED"]:
            break
        wait = float(response.headers.get("Retry-After", 10))
        time.sleep(wait)
    children = data.get("children", [])
    if not children or data.get("status") in {"ERROR", "FAILED", "CANCELLED"}:
        return {"completed": False, "result": []}
    results = []
    for child in children:
        child_id = child.get("id") if isinstance(child, dict) else child
        if not child_id:
            logger.warning("Skipping malformed multi-simulation child: %r", child)
            continue
        child_response = s.get(
            f"{brain_api_url}/simulations/{child_id}", timeout=15
        )
        if child_response.status_code // 100 != 2:
            logger.warning(
                "Could not read multi-simulation child %s (HTTP %s).",
                child_id,
                child_response.status_code,
            )
            continue
        try:
            child_info = child_response.json()
        except Exception:
            logger.warning("Child %s returned non-JSON data.", child_id)
            continue
        alpha_value = child_info.get("alpha")
        alpha_id = (
            alpha_value.get("id")
            if isinstance(alpha_value, dict)
            else alpha_value
        )
        if alpha_id:
            alpha_result = get_simulation_result_json(s, alpha_id)
            if not isinstance(alpha_result, dict):
                alpha_result = {}
            alpha_result.setdefault("id", alpha_id)
            results.append(alpha_result)
        else:
            logger.warning("No alpha ID was returned for child %s.", child_id)
    return {"completed": True, "result": results}

def simulate_single_alpha(s: SingleSession, simulate_data: dict) -> dict:
    if check_session_timeout(s) <= 0:
        if not verify_session_truth_silent(s):
            raise SessionExpiredError("Session dead before starting simulation.")
    resp = s.post(f"{brain_api_url}/simulations", json=simulate_data)
    if resp.status_code == 401:
        if verify_session_truth_silent(s):
            return simulate_single_alpha(s, simulate_data)
        raise SessionExpiredError("Authentication failed at simulation start.")
    res = simulation_progress(s, resp)
    if res["completed"]:
        alpha_id = res["result"].get("id")
        if alpha_id:
            try:
                set_alpha_properties(s, alpha_id)
            except Exception as exc:
                logger.warning(
                    "Alpha %s was created but could not be updated: %s",
                    alpha_id,
                    exc,
                )
            return {"alpha_id": alpha_id, "simulate_data": simulate_data}
    expression = simulate_data.get("regular") or simulate_data.get("selection")
    logger.error("BRAIN rejected or failed this expression: %s", expression)
    return {"alpha_id": None, "simulate_data": simulate_data}

def _retry_multi_batch_individually(
    s: SingleSession,
    simulate_data_list: list,
    reason: str,
) -> list[dict]:
    logger.warning(
        "Multi-simulation batch produced no usable IDs (%s). "
        "Retrying its %s expressions one by one so invalid expressions are isolated.",
        reason,
        len(simulate_data_list),
    )
    recovered = []
    for index, simulate_data in enumerate(simulate_data_list, start=1):
        expression = simulate_data.get("regular") or simulate_data.get("selection")
        logger.info(
            "Individual fallback %s/%s: %s",
            index,
            len(simulate_data_list),
            expression,
        )
        recovered.append(simulate_single_alpha(s, simulate_data))
    return recovered

def simulate_multi_alpha(s: SingleSession, simulate_data_list: list) -> list[dict]:
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
    if check_session_timeout(s) <= 0:
        if not verify_session_truth_silent(s):
            raise SessionExpiredError("Session dead before starting multi-simulation.")
    resp = s.post(f"{brain_api_url}/simulations", json=simulate_data_list)
    if resp.status_code == 401:
        if verify_session_truth_silent(s):
            return simulate_multi_alpha(s, simulate_data_list)
        raise SessionExpiredError("Authentication failed at multi-simulation start.")
    if resp.status_code // 100 != 2:
        logger.error(
            "Multi-simulation request failed (HTTP %s): %s",
            resp.status_code,
            resp.text[:500],
        )
        return _retry_multi_batch_individually(
            s,
            simulate_data_list,
            f"batch HTTP {resp.status_code}",
        )
    res = multisimulation_progress(s, resp)
    if not res["completed"]:
        return _retry_multi_batch_individually(
            s,
            simulate_data_list,
            "terminal batch error",
        )
    output = []
    for index, x in enumerate(res["result"]):
        if not isinstance(x, dict):
            logger.warning("Skipping malformed alpha result: %r", x)
            continue
        alpha_id = x.get("id")
        if not alpha_id:
            continue
        returned_regular = x.get("regular") or {}
        returned_code = (
            returned_regular.get("code")
            if isinstance(returned_regular, dict)
            else returned_regular
        )
        sim_data = next(
            (
                item for item in simulate_data_list
                if item.get("regular") == returned_code
            ),
            simulate_data_list[min(index, len(simulate_data_list) - 1)],
        )
        output.append({"alpha_id": alpha_id, "simulate_data": sim_data})
        try:
            set_alpha_properties(s, alpha_id)
        except Exception as exc:
            logger.warning("Alpha %s was created but could not be updated: %s", alpha_id, exc)
    if not output:
        return _retry_multi_batch_individually(
            s,
            simulate_data_list,
            "completed batch returned no alpha IDs",
        )
    recovered_simulations = [item["simulate_data"] for item in output]
    missing = [
        item for item in simulate_data_list
        if item not in recovered_simulations
    ]
    if missing:
        output.extend(
            _retry_multi_batch_individually(
                s,
                missing,
                "some batch children failed",
            )
        )
    return output

def simulate_alpha_list_multi(
    s: SingleSession,
    alpha_list: list,
    limit_of_concurrent_simulations: int = 6,
    limit_of_multi_simulations: int = 10,
    simulation_config: dict = DEFAULT_CONFIG
) -> list:
    if not alpha_list:
        raise ValueError(
            "No alpha configurations were generated. Check that the templates list "
            "contains at least one active expression."
        )
    invalid_items = [item for item in alpha_list if not isinstance(item, dict) or not item]
    if invalid_items:
        raise ValueError("alpha_list contains an invalid or empty simulation configuration.")
    batch_size = max(1, min(int(limit_of_multi_simulations), 10))
    concurrency = max(1, min(int(limit_of_concurrent_simulations), 8))
    start_time_total = time.time()
    s = check_session_and_relogin(s)
    remaining_time = check_session_timeout(s)
    if remaining_time <= 0:
        logger.error("Simulation aborted: Session is already expired.")
        return []
    tasks = [alpha_list[i : i + batch_size] for i in range(0, len(alpha_list), batch_size)]
    logger.info(f"FACTORY SPINNING UP: {len(alpha_list)} alphas | {len(tasks)} batches | {concurrency} parallel streams")
    flat_results = []
    try:
        with ThreadPool(concurrency) as pool:
            with tqdm.tqdm(total=len(tasks), desc="Simulating") as pbar:
                iterator = pool.imap_unordered(partial(simulate_multi_alpha, s), tasks)
                while True:
                    try:
                        batch_start = time.time()
                        batch = next(iterator)
                        batch_duration = time.time() - batch_start
                        flat_results.extend(batch)
                        logger.debug(f"Batch Success: {len(batch)} IDs recovered in {batch_duration:.2f}s")
                        pbar.update()
                    except StopIteration:
                        break
    except SessionExpiredError:
        logger.warning(f"Session expired mid-simulation. Progress halted. Recovered {len(flat_results)} alphas so far.")
    except Exception as e:
        logger.error(f"Critical Simulation Error: {str(e)}")
    if not flat_results:
        logger.warning("No alpha IDs were recovered from the simulation phase.")
        return []
    logger.info(f"SIMULATIONS COMPLETE: {len(flat_results)} alphas ready | Fetching metrics now...")
    valid_results = [
        item for item in flat_results
        if isinstance(item, dict) and item.get("alpha_id")
    ]
    if not valid_results:
        raise RuntimeError(
            "BRAIN returned no alpha IDs. Review the HTTP error messages above; "
            "no simulations were recovered."
        )
    stats_output = []
    stats_start_time = time.time()
    with ThreadPool(3) as pool:
        def fetcher(item):
            try:
                return get_specified_alpha_stats(
                    s,
                    item["alpha_id"],
                    item["simulate_data"],
                    **simulation_config,
                )
            except Exception as exc:
                logger.warning(
                    "Alpha %s exists on BRAIN, but its metrics could not be loaded: %s",
                    item["alpha_id"],
                    exc,
                )
                return {
                    "alpha_id": item["alpha_id"],
                    "simulate_data": item["simulate_data"],
                    "is_stats": None,
                    "pnl": None,
                    "stats": None,
                    "is_tests": None,
                    "train": None,
                    "test": None,
                }
        for stat_res in pool.map(fetcher, valid_results):
            if stat_res:
                stats_output.append(stat_res)
    total_duration = time.time() - start_time_total
    stats_duration = time.time() - stats_start_time
    logger.info(f"BATCH COMPLETE: {len(stats_output)} alphas locked & loaded | Total time: {total_duration:.2f}s (Stats: {stats_duration:.2f}s)")
    return _delete_duplicates_from_result(stats_output)

def get_specified_alpha_stats(
    s: SingleSession,
    alpha_id: Union[str, None],
    simulate_data: dict,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
) -> dict:
    pnl = None
    stats = None
    logger.debug(f"Session (ID: {id(s)}) used in get_specified_alpha_stats for alpha_id: {alpha_id}")
    if alpha_id is None:
        return {
            "alpha_id": None,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": pnl,
            "stats": stats,
            "is_tests": None,
            "train": None,
            "test": None,
        }
    result = get_simulation_result_json(s, alpha_id)
    try:
        region = result["settings"]["region"]
        is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key != 'checks'}]).assign(
            alpha_id=alpha_id
        )
    except Exception as e:
        logger.error(f"Failed to retrieve simulation result for alpha_id {alpha_id}: {result}, {e}")
    train = result["train"]
    test = result["test"]
    is_stats = pd.DataFrame([{key: value for key, value in result["is"].items() if key != "checks"}]).assign(
        alpha_id=alpha_id
    )
    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
        if save_pnl_file:
            save_pnl(pnl, alpha_id, region)
    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)
        if save_stats_file:
            save_yearly_stats(stats, alpha_id, region)
    if save_result_file:
        save_simulation_result(result)
    is_tests = pd.DataFrame(result["is"]["checks"]).assign(alpha_id=alpha_id)
    if check_submission:
        is_tests = get_check_submission(s, alpha_id)
        return {
            "alpha_id": alpha_id,
            "simulate_data": simulate_data,
            "is_stats": is_stats,
            "pnl": pnl,
            "stats": stats,
            "is_tests": is_tests,
            "train": train,
            "test": test,
        }
    if check_self_corr and not check_submission:
        self_corr_test = check_self_corr_test(s, alpha_id)
        is_tests = (
            pd.concat([is_tests, pd.DataFrame([self_corr_test])], ignore_index=True)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    if check_prod_corr and not check_submission:
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        is_tests = (
            pd.concat([is_tests, pd.DataFrame([prod_corr_test])], ignore_index=True)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    return {
        "alpha_id": alpha_id,
        "simulate_data": simulate_data,
        "is_stats": is_stats,
        "pnl": pnl,
        "stats": stats,
        "is_tests": is_tests,
        "train": train,
        "test": test,
    }

def _delete_duplicates_from_result(result: list) -> list:
    alpha_id_lst = []
    result_new = []
    for x in result:
        if not isinstance(x, dict) or not x:
            continue
        alpha_id = x.get("alpha_id")
        if alpha_id is not None:
            if alpha_id not in alpha_id_lst:
                result_new.append(x)
                alpha_id_lst.append(alpha_id)
    return result_new

def get_prod_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/prod")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get production correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    prod_corr_df["alpha_max_prod_corr"] = result.json()["max"]
    prod_corr_df["alpha_min_prod_corr"] = result.json()["min"]
    return prod_corr_df

def check_prod_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    prod_corr_df = get_prod_corr(s, alpha_id)
    if prod_corr_df.empty:
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "NONE",
                "limit": threshold,
                "value": None,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "PASS" if value <= threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)

def get_self_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/self")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get self correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()
    records_len = len(result.json()["records"])
    if records_len == 0:
        logger.warning(f"No self correlation for alpha_id {alpha_id}")
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    self_corr_df["alpha_max_self_corr"] = result.json()["max"]
    self_corr_df["alpha_min_self_corr"] = result.json()["min"]
    return self_corr_df

def check_self_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS",
                "limit": threshold,
                "value": 0,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)

def get_check_submission(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        logger.warning(f"Cant check submission alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()
    checks_df = pd.DataFrame(result.json()["is"]["checks"]).assign(alpha_id=alpha_id)
    return checks_df

def generate_alpha(
    regular: Optional[str] = None,
    selection: Optional[str] = None,
    combo: Optional[str] = None,
    alpha_type: Literal["REGULAR", "SUPER"] = "REGULAR",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: Literal[0, 1] = 1,
    decay: int = 0,
    neutralization: str = "SUBINDUSTRY",
    truncation: float = 0.08,
    pasteurization: Literal["ON", "OFF"] = "ON",
    test_period: str = "P2Y0M0D",
    unit_handling: Literal["VERIFY"] = "VERIFY",
    nan_handling: Literal["ON", "OFF"] = "OFF",
    max_trade: Literal["ON", "OFF"] = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 20,
    visualization: bool = False,
) -> dict:
    settings = {
        "instrumentType": "EQUITY",
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "testPeriod": test_period,
        "unitHandling": unit_handling,
        "nanHandling": nan_handling,
        "maxTrade": max_trade,
        "language": "FASTEXPR",
        "visualization": visualization,
    }
    if alpha_type == "REGULAR":
        simulation_data = {
            "type": alpha_type,
            "settings": settings,
            "regular": regular,
        }
    elif alpha_type == "SUPER":
        simulation_data = {
            "type": alpha_type,
            "settings": {
                **settings,
                "selectionHandling": selection_handling,
                "selectionLimit": selection_limit,
            },
            "combo": combo,
            "selection": selection,
        }
    else:
        logger.error("alpha_type should be REGULAR or SUPER")
        return {}
    return simulation_data

regular_description =  """
Idea: 
Define the core anomaly and strategy logic.

Rationale for data used: 
Explain why these fields predict future returns.

Rationale for operators used: 
Detail how math functions isolate the signal.
"""
combo_desc = '''For each selection, one of two Combo Expressions was used to generate SuperAlphas:
1. Equal-weighted: This combo expression assigns equal weight to each Alpha, ensuring that all Alphas contribute equally to the overall performance of the SuperAlpha.
2. Information ratio-weighted: This combo expression weights each Alpha by its information ratio calculated from the previous 500 trading days (~2 years). 
    This approach prioritizes Alphas with higher information ratios, potentially enhancing the performance of the SuperAlpha.

Additionally, because Alphas with turnover greater than 0.5 were selected, a Decay of 10 was used. 
This decay factor helps to manage the impact of Alphas over time, ensuring that more recent data has a greater influence. 
Neutralization to Fast Factors was also applied to mitigate the impact of fast-moving market factors.
'''
selection_desc = '''Selection Expression template description.
- in(datacategories, "{category}"):
This condition checks if the data belongs to a specified category. The {category} placeholder will be replaced with actual category IDs.
- datafield_count<=3:
This ensures that only Alphas with 3 or less data fields are selected.
- turnover>={turnover_l}:
This condition filters entries where the turnover is greater than or equal to a lower threshold ({turnover_l}). This placeholder will be replaced with actual values in range(0.50, 0.61).
- turnover<{turnover_r}:
This filters entries where the turnover is less than an upper threshold ({turnover_r}). This placeholder will also be replaced with actual values in range(0.51, 0.62).
- os_start_date>"2020-01-01":
This ensures that only Alphas with a OS start date later than January 1, 2020, are selected.'''

def set_alpha_properties(
    s: SingleSession,
    alpha_id: str,
    name: Optional[str] = 'api_alpha',
    color: Optional[str] = None,
    category: Optional[
        Literal[
            "PRICE_REVERSION",
            "PRICE_MOMENTUM",
            "VOLUME",
            "FUNDAMENTAL",
            "ANALYST",
            "PRICE_VOLUME",
            "RELATION",
            "SENTIMENT",
        ]
    ] = None,
    regular_desc: Optional[str] = regular_description,
    selection_desc: Optional[str] = selection_desc,
    combo_desc: Optional[str] = combo_desc,
    osmosis_points: Optional[int] = None,
    tags: Optional[list[str]] = ['api'],
) -> requests.Response:
    if osmosis_points is not None and not (1 <= osmosis_points <= 100_000):
        raise ValueError(f"osmosis_points must be between 1 and 100000, got {osmosis_points}")
    option_map = {
        "name": name,
        "color": color,
        "tags": tags,
        "category": category,
        "osmosisPoints": osmosis_points,
        "regular": {"description": regular_desc} if regular_desc is not None else None,
        "selection": {"description": selection_desc} if selection_desc is not None else None,
        "combo": {"description": combo_desc} if combo_desc is not None else None,
    }
    params = {k: v for k, v in option_map.items() if v is not None}
    response = s.patch(brain_api_url + "/alphas/" + alpha_id, json=params)
    return response

def _get_alpha_pnl(
    s: SingleSession,
    alpha_id: str,
    pnl_type: str = "pnl",
) -> pd.DataFrame:
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + f"/recordsets/{pnl_type}")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json()
    if pnl.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in pnl["schema"]["properties"]]
    pnl_df = (
        pd.DataFrame(pnl["records"], columns=columns)
        .assign(alpha_id=alpha_id, date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"))
        .set_index("date")
    )
    return pnl_df

def get_alpha_pnl(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    return _get_alpha_pnl(s, alpha_id, "pnl")

def get_alpha_yearly_stats(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/recordsets/yearly-stats")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    stats = result.json()
    if stats.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in stats["schema"]["properties"]]
    yearly_stats_df = pd.DataFrame(stats["records"], columns=columns).assign(alpha_id=alpha_id)
    return yearly_stats_df

def _check_rate_limit(response: requests.Response) -> None:
    header_keys = {
        "limit_minute": "x-ratelimit-limit-minute",
        "remaining_minute": "x-ratelimit-remaining-minute",
        "limit_second": "x-ratelimit-limit-second",
        "remaining_second": "x-ratelimit-remaining-second",
    }
    parsed = {}
    for key, header_name in header_keys.items():
        val = response.headers.get(header_name)
        if val is None:
            logger.warning(f"Failed to parse rate-limit values: missing header {header_name}")
            return
        try:
            parsed[key] = int(val)
        except (ValueError, TypeError):
            parsed[key] = 30
            logger.warning(f"Failed to parse rate-limit values: cannot convert {header_name}={val} to int")
            return
    if parsed["remaining_second"] < 1:
        time.sleep(1)
    if parsed["remaining_minute"] <= 1:
        logger.info(f"Rate limit {parsed['limit_minute']} reached (per minute). Sleeping for a minute...")
        time.sleep(60)

def get_datasets(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: Optional[bool] = None,
) -> pd.DataFrame:
    url = (
        brain_api_url
        + "/data-sets?"
        + f"instrumentType={instrument_type}&region={region}&delay={delay}&universe={universe}"
    )
    if theme is not None:
        theme_str = "true" if theme else "false"
        url += f"&theme={theme_str}"
    result = s.get(url)
    _check_rate_limit(result)
    datasets_df = pd.DataFrame(result.json()["results"])
    datasets_df = expand_dict_columns(datasets_df)
    return datasets_df

def get_datafields(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: Optional[bool] = None,
    dataset_id: str = "",
    data_type: str = "MATRIX",
    search: str = "",
) -> pd.DataFrame:
    type_param = f"&type={data_type}" if data_type != "ALL" else ""
    if len(search) == 0:
        url_template = (
            brain_api_url
            + "/data-fields?"
            + f"&instrumentType={instrument_type}"
            + f"&region={region}&delay={str(delay)}&universe={universe}{type_param}&dataset.id={dataset_id}&limit=50"
            + "&offset={x}"
        )
        count = s.get(url_template.format(x=0)).json()["count"]
        if count == 0:
            logger.warning(
                f"No fields found: region={region}, delay={str(delay)}, universe={universe}, "
                f"type={data_type}, dataset.id={dataset_id}"
            )
            return pd.DataFrame()
    else:
        url_template = (
            brain_api_url
            + "/data-fields?"
            + f"&instrumentType={instrument_type}"
            + f"&region={region}&delay={str(delay)}&universe={universe}{type_param}&limit=50"
            + f"&search={search}"
            + "&offset={x}"
        )
        count = 100
    max_try = 5
    datafields_list = []
    for x in range(0, count, 50):
        for _ in range(max_try):
            datafields = s.get(url_template.format(x=x))
            if "results" in datafields.json():
                break
            time.sleep(5)
        datafields_list.append(datafields.json()["results"])
    datafields_list_flat = [item for sublist in datafields_list for item in sublist]
    datafields_df = pd.DataFrame(datafields_list_flat)
    datafields_df = expand_dict_columns(datafields_df)
    return datafields_df

def get_operators(s: SingleSession) -> pd.DataFrame:
    df = pd.DataFrame(s.get(brain_api_url + "/operators").json())
    return df.explode('scope').reset_index(drop=True)

def get_instrument_type_region_delay(s: SingleSession) -> pd.DataFrame:
    settings_options = s.options(brain_api_url + '/simulations').json()['actions']['POST']['settings']['children']
    data = [
        {settings_options[key]['label']: settings_options[key]['choices']}
        for key in settings_options.keys()
        if settings_options[key]['type'] == 'choice'
    ]
    instrument_type_data = {}
    region_data = {}
    universe_data = {}
    delay_data = {}
    neutralization_data = {}
    for item in data:
        if 'Instrument type' in item:
            instrument_type_data = item['Instrument type']
        elif 'Region' in item:
            region_data = item['Region']['instrumentType']
        elif 'Universe' in item:
            universe_data = item['Universe']['instrumentType']
        elif 'Delay' in item:
            delay_data = item['Delay']['instrumentType']
        elif 'Neutralization' in item:
            neutralization_data = item['Neutralization']['instrumentType']
    data_list = []
    for instrument_type in instrument_type_data:
        for region in region_data[instrument_type['value']]:
            for delay in delay_data[instrument_type['value']]['region'][region['value']]:
                row = {'InstrumentType': instrument_type['value'], 'Region': region['value'], 'Delay': delay['value']}
                row['Universe'] = [
                    item['value'] for item in universe_data[instrument_type['value']]['region'][region['value']]
                ]
                row['Neutralization'] = [
                    item['value'] for item in neutralization_data[instrument_type['value']]['region'][region['value']]
                ]
                data_list.append(row)
    df = (
        pd.DataFrame(data_list)
        .sort_values(
            by=['InstrumentType', 'Region', 'Delay'],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return df

def performance_comparison(
    s: SingleSession, alpha_id: str, team_id: Optional[str] = None, competition: Optional[str] = None
) -> dict:
    if competition is not None:
        part_url = f"competitions/{competition}"
    elif team_id is not None:
        part_url = f"teams/{team_id}"
    else:
        part_url = "users/self"
    while True:
        result = s.get(brain_api_url + f"/{part_url}/alphas/" + alpha_id + "/before-and-after-performance")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("stats", 0) == 0:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}
    if result.status_code != 200:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}
    return result.json()

def construct_selection_expression(
    selection: str,
    instrument_type: Literal["EQUITY"] = "EQUITY",
    region: str = "USA",
    delay: Literal[0, 1] = 1,
    selection_limit: int = 10,
    selection_handling: str = "POSITIVE",
) -> dict:
    selection_data = {
        "settings.instrumentType": instrument_type,
        "settings.region": region,
        "settings.delay": delay,
        "selection": selection,
        "limit": 10,
        "selectionLimit": selection_limit,
        "selectionHandling": selection_handling,
    }
    return selection_data

def run_selection(s: SingleSession, selection_data: dict) -> dict:
    selection_response = s.get(brain_api_url + "/simulations/super-selection", params=selection_data)
    r = selection_response.json()
    selected_alphas_count = r.get("count")
    message = r.get("message", "")
    time.sleep(2)
    return {"selected_alphas_count": selected_alphas_count, "message": message}

def markandstore(result, s, dataset, Region):
    result = [
    x for x in result 
    if isinstance(x, dict) and x.get('alpha_id') is not None
    ]
    print(f"Cleaned! Usable alphas with IDs: {len(result)}")
    stats_list = []
    for x in result:
        stats_list.append({
            'alpha_id': x['alpha_id'],
            'fitness': x['is_stats']['fitness'].item(),
            'sharpe': x['is_stats']['sharpe'].item()
        })
    df_stats = pd.DataFrame(stats_list)
    if not df_stats.empty:
        top_5_fitness = df_stats.nlargest(5, 'fitness')['alpha_id'].tolist()
        top_5_sharpe = df_stats.nlargest(5, 'sharpe')['alpha_id'].tolist()
    else:
        top_5_fitness = []
        top_5_sharpe = []
    yellow_ids = set(top_5_fitness + top_5_sharpe)
    passThreshold = int(result[0]['is_tests']['limit'].count())+1
    print(f"Pass Threshold set to: {passThreshold} (Total Tests: {passThreshold - 1})")
    green_alphas = []
    yellow_alphas = []
    red_alphas = []
    purple_alphas  = []
    for x in result:
        is_tests_df = x['is_tests']
        alpha_id = x['alpha_id']
        expression = x['simulate_data']['regular']
        pass_count = (is_tests_df['result'] == 'PASS').sum()
        has_fail = (is_tests_df['result'] == 'FAIL').any()
        if x['is_stats']['sharpe'].item() > 2.0 and has_fail:
            set_alpha_properties(s, alpha_id, color='PURPLE', name=dataset, tags=['top_performer', 'ace_tag'])
            expression = x['simulate_data']['regular']
            purple_alphas.append(expression)
        if alpha_id in yellow_ids and has_fail and x['is_stats']['sharpe'].item() > 1.5:
            set_alpha_properties(s, alpha_id, color='YELLOW',  name=dataset, tags=['top_performer', 'ace_tag'])
            expression = x['simulate_data']['regular']
            yellow_alphas.append(expression)
        if  not has_fail and pass_count > passThreshold :
            expression = x['simulate_data']['regular']
            green_alphas.append(expression)
            set_alpha_properties(
            s, 
            alpha_id, 
            color='GREEN', 
            name=dataset,
            tags=['ace_tag', '_Toothless']
            )
            stats = x['is_stats']['investabilityConstrained'].iloc[0]
            returns = stats['returns']
            drawdown = stats['drawdown']
            alpha_id = x['alpha_id']
            ret_to_dd = returns / drawdown if drawdown != 0 else 0
            if ret_to_dd > 1:
                blue_count += 1 
                print(f"Target Met! Alpha {alpha_id} Ratio: {ret_to_dd:.2f}")
                set_alpha_properties(
                    s, 
                    alpha_id, 
                    color='BLUE', 
                    name=dataset,
                    tags=['ace_tag', '_Toothless']
                )
        if x['is_stats']['sharpe'].item() < -1.7:
            set_alpha_properties(s, alpha_id, color='RED',   tags=['low_performer', 'ace_tag'])
            expression = x['simulate_data']['regular']
            red_alphas.append(expression)
    COUNT = 0
    print("green Alphas")
    for i in range(len(green_alphas)):
        COUNT += 1
    print(COUNT)
    COUNT = 0
    print("YhELLLO Alphas")
    for i in range(len(yellow_alphas)):
        COUNT += 1
    print(COUNT)
    COUNT = 0 
    print("RED Alphas")
    for i in range(len(red_alphas)):
        COUNT += 1
    print(COUNT)    
    extracted_expressions = list(dict.fromkeys(yellow_alphas))
    timestamp_key = datetime.datetime.now().strftime("%H:%M:%S")
    list_name = f"{dataset}{Region}@{timestamp_key}YELLOW"
    master_file = "yellow_alphas.json"
    if os.path.exists(master_file):
        with open(master_file, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}
    all_data[list_name] = extracted_expressions
    with open(master_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Added {len(extracted_expressions)} alphas to '{master_file}' under name: {list_name}")
    extracted_expressions = list(dict.fromkeys(green_alphas))
    timestamp_key = datetime.datetime.now().strftime("%H:%M:%S")
    list_name = f"{dataset}{Region}@{timestamp_key}GREEN"
    master_file = "green_alphas.json"
    if os.path.exists(master_file):
        with open(master_file, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}
    all_data[list_name] = extracted_expressions
    with open(master_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Added {len(extracted_expressions)} alphas to '{master_file}' under name: {list_name}")    
    
    extracted_expressions = list(dict.fromkeys(purple_alphas))
    timestamp_key = datetime.datetime.now().strftime("%H:%M:%S")
    list_name = f"{dataset}{Region}@{timestamp_key}Purplez"
    master_file = "purple_alphas.json"
    if os.path.exists(master_file):
        with open(master_file, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}
    all_data[list_name] = extracted_expressions
    with open(master_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Added {len(extracted_expressions)} alphas to '{master_file}' under name: {list_name}")    
    return None

def XhinkuAlphaFilter(result, s, dataset):
    result = [
    x for x in result 
    if isinstance(x, dict) and x.get('alpha_id') is not None
    ]
    print(f"Cleaned! Usable alphas with IDs: {len(result)}")
    if len(result) == 0:
        print("No valid alphas found in the result. Exiting function.")
        return None
    stats_list = []
    for x in result:
        stats_list.append({
            'alpha_id': x['alpha_id'],
            'fitness': x['is_stats']['fitness'].item(),
            'sharpe': x['is_stats']['sharpe'].item()
        })
    df_stats = pd.DataFrame(stats_list)
    if not df_stats.empty:
        top_5_fitness = df_stats.nlargest(5, 'fitness')['alpha_id'].tolist()
        top_5_sharpe = df_stats.nlargest(5, 'sharpe')['alpha_id'].tolist()
    else:
        top_5_fitness = []
        top_5_sharpe = []
    yellow_ids = set(top_5_fitness + top_5_sharpe)
    green_alphas = []
    yellow_alphas = []
    red_alphas = []
    purple_alphas  = []
    passThreshold = int(result[0]['is_tests']['limit'].count())+1
    for x in result:
        is_tests_df = x['is_tests']
        alpha_id = x['alpha_id']
        expression = x['simulate_data']['regular']
        pass_count = (is_tests_df['result'] == 'PASS').sum()
        has_fail = (is_tests_df['result'] == 'FAIL').any()
        if x['is_stats']['sharpe'].item() > 2.0 and has_fail:
            set_alpha_properties(s, alpha_id, color='PURPLE', name=dataset, tags=['mid',])
            expression = x['simulate_data']['regular']
            purple_alphas.append(expression)
        if alpha_id in yellow_ids and has_fail and x['is_stats']['sharpe'].item() > 1.5:
            set_alpha_properties(s, alpha_id, color='YELLOW',  name=dataset, tags=['mid',])
            expression = x['simulate_data']['regular']
            yellow_alphas.append(expression)
        if  not has_fail and pass_count > passThreshold :
            expression = x['simulate_data']['regular']
            green_alphas.append(expression)
            set_alpha_properties(
            s, 
            alpha_id, 
            color='GREEN', 
            name=dataset,
            tags=[ 'GoodAlpha']
            )
            stats = x['is_stats']['investabilityConstrained'].iloc[0]
            returns = stats['returns']
            drawdown = stats['drawdown']
            alpha_id = x['alpha_id']
            ret_to_dd = returns / drawdown if drawdown != 0 else 0
            if ret_to_dd > 1:
                print(f"Target Met! Alpha {alpha_id} Ratio: {ret_to_dd:.2f}")
                set_alpha_properties(
                    s, 
                    alpha_id, 
                    color='BLUE', 
                    name=dataset,
                    tags=[ 'GoodAlpha']
                )
        if x['is_stats']['sharpe'].item() < -1.7:
            set_alpha_properties(s, alpha_id, color='RED',   tags=['baddie',])
            expression = x['simulate_data']['regular']
            red_alphas.append(expression)
    COUNT = 0
    print("green Alphas")
    for i in range(len(green_alphas)):
        COUNT += 1
    print(COUNT)
    COUNT = 0
    print("YhELLLO Alphas")
    for i in range(len(yellow_alphas)):
        COUNT += 1
    print(COUNT)
    COUNT = 0 
    print("RED Alphas")
    for i in range(len(red_alphas)):
        COUNT += 1
    print(COUNT)
    COUNT = 0
    print("PURPLE Alphas")
    for i in range(len(purple_alphas)):
        COUNT += 1
    print(COUNT)
    
import time
def submit_alpha(s, alpha_id):
    time.sleep(10)
    checks_df = get_check_submission(s, alpha_id)
    failures = checks_df.query('result == "FAIL"')
    if not failures.empty:
        failed_names = ", ".join(failures['name'].tolist())
        msg = f"Pre-submission checks failed: {failed_names}"
        if failed_names == 'REGULAR_SUBMISSION' or failed_names == 'SUPER_SUBMISSION':
            raise ValueError(f"Daily submission limit reached. Try Again tomorrow.")
        else:
            set_alpha_properties(s, alpha_id, tags=['_Beta'])
            return False, msg
    try:
        print(f"[{alpha_id}] Step 2: Signal is clean. Handing over to Platform for final submission...")
        result = s.post(f"{brain_api_url}/alphas/{alpha_id}/submit")
        while "retry-after" in result.headers:
            wait_time = float(result.headers["Retry-After"])
            print(f"   > Platform is processing... (waiting {wait_time}s)", end="\r")
            time.sleep(wait_time)
            result = s.get(f"{brain_api_url}/alphas/{alpha_id}/submit")
        if result.status_code == 200:
            print(f"[{alpha_id}] Step 3: Platform verification complete. REAL SUBMISSION SUCCESS.          ")
            return True, "Submitted successfully"
        else:
            print(f"[{alpha_id}] Step 3: Platform REJECTED the submission.                               ")
            set_alpha_properties(s, alpha_id, color="YELLOW",)
            return False, f"Platform rejected (Status {result.status_code})"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def submit_alpha_list(s, alpha_id_list, daily_limit=4):
    count = 0
    print(f"Starting submission process for {len(alpha_id_list)} alphas...")
    for id in alpha_id_list:
        if count >= daily_limit:
            print(f"\nDaily limit of {daily_limit} reached. Stopping for the day.")
            break
        print(f"Checking {id}...", end=" ", flush=True)
        is_success, message = submit_alpha(s, id)
        if is_success:
            print(f"{message}")
            count += 1
        else:
            print(f"{message}")
    print(f"\n Batch complete. Total Alphas Submitted today: {count}")