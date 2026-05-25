"""
src/utils/config.py
===================
Environment-first configuration loader.

The project uses a .env-style file for deploy-time settings because it maps
cleanly to shell variables, containers, and CI secrets. Values from the real
process environment always override values in the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load .env-style config and return the nested dict used by the app."""
    file_values = _read_env_file(Path(config_path))

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name, file_values.get(name, default))

    log_file = get("LOG_FILE", "logs/agent.log")
    log_level = get("LOG_LEVEL", "INFO").upper()
    debug_level = get("LOG_FILE_LEVEL", "DEBUG").upper()

    return {
        "budget": {
            "daily_limit": _as_int(get("BUDGET_DAILY_LIMIT", "50")),
            "max_steps": _as_int(get("BUDGET_MAX_STEPS", "20")),
            "generator_cost": _as_int(get("BUDGET_GENERATOR_COST", "1")),
            "critic_cost": _as_int(get("BUDGET_CRITIC_COST", "1")),
        },
        "ucb": {
            "c": _as_float(get("UCB_C", "1.5")),
            "c_decay": _as_float(get("UCB_C_DECAY", "0.95")),
            "min_c": _as_float(get("UCB_MIN_C", "0.3")),
        },
        "evaluator": {
            "iv_threshold": _as_float(get("EVAL_IV_THRESHOLD", "0.005")),
            "ks_threshold": _as_float(get("EVAL_KS_THRESHOLD", "0.01")),
            "missing_rate_threshold": _as_float(
                get("EVAL_MISSING_RATE_THRESHOLD", "0.7")
            ),
            "psi_threshold": _as_float(get("EVAL_PSI_THRESHOLD", "0.25")),
            "n_bins": _as_int(get("EVAL_N_BINS", "10")),
        },
        "generator": {
            "k_features_per_step": _as_int(get("GENERATOR_K_FEATURES_PER_STEP", "3")),
            "temperature": _as_float(get("GENERATOR_TEMPERATURE", "0.7")),
            "max_tokens": _as_int(get("GENERATOR_MAX_TOKENS", "2048")),
            "model": get("GENERATOR_MODEL", get("LLM_MODEL", "deepseek-chat")),
        },
        "critic": {
            "temperature": _as_float(get("CRITIC_TEMPERATURE", "0.3")),
            "max_tokens": _as_int(get("CRITIC_MAX_TOKENS", "1024")),
            "model": get("CRITIC_MODEL", get("LLM_MODEL", "deepseek-chat")),
            "failure_threshold": _as_float(get("CRITIC_FAILURE_THRESHOLD", "0.2")),
        },
        "llm": {
            "mode": get("LLM_MODE", "api"),
            "api": {
                "provider": get("LLM_PROVIDER", "deepseek"),
                "base_url": get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
                "api_key": _first_non_empty(
                    get("LLM_API_KEY", ""),
                    os.environ.get("DEEPSEEK_API_KEY", ""),
                    os.environ.get("OPENAI_API_KEY", ""),
                ),
                "model": get("LLM_MODEL", "deepseek-chat"),
                "timeout": _as_int(get("LLM_TIMEOUT", "60")),
                "max_retries": _as_int(get("LLM_MAX_RETRIES", "3")),
                "allow_mock_fallback": _as_bool(
                    get("LLM_ALLOW_MOCK_FALLBACK", "false")
                ),
                "temperature_generator": _as_float(
                    get("GENERATOR_TEMPERATURE", "0.7")
                ),
                "temperature_critic": _as_float(get("CRITIC_TEMPERATURE", "0.3")),
            },
        },
        "data": {
            "source": get("DATA_SOURCE", "home_credit"),
            "home_credit_dir": get("HOME_CREDIT_DIR", ""),
        },
        "run": {
            "base_dir": get("RUN_BASE_DIR", "runs"),
        },
        "strategy": {
            "template_families": _as_list(
                get(
                    "TEMPLATE_FAMILIES",
                    "time_window_agg,ratio_share,volatility,cross_feature,"
                    "anomaly_distance,stability_shift",
                )
            ),
            "transform_strategies": _as_list(
                get(
                    "TRANSFORM_STRATEGIES",
                    "raw,log,woe_bin,quantile_bin,missing_indicator,interaction",
                )
            ),
        },
        "knowledge": {
            "feature_bank_path": get(
                "FEATURE_BANK_PATH", "knowledge/feature_bank.json"
            ),
        },
        "trace": {
            "enabled": _as_bool(get("TRACE_ENABLED", "true")),
        },
        "directions": _as_list(
            get(
                "DIRECTIONS",
                "user_device_risk,transaction_pattern,geolocation_anomaly,"
                "account_behavior,network_relation",
            )
        ),
        "compliance": {
            "required_tags": _as_list(
                get("COMPLIANCE_REQUIRED_TAGS", "PII_FREE,EXPLAINABLE")
            ),
            "forbidden_fields": _as_list(
                get("COMPLIANCE_FORBIDDEN_FIELDS", "id_number,phone,name,email_raw")
            ),
        },
        "logging": _build_logging_config(log_level, debug_level, log_file),
    }


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = _strip_quotes(value.strip())
    return values


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _as_int(value: str) -> int:
    return int(value.strip())


def _as_float(value: str) -> float:
    return float(value.strip())


def _as_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _build_logging_config(log_level: str, file_level: str, log_file: str) -> dict:
    return {
        "version": 1,
        "formatters": {
            "json": {
                "format": (
                    '{"time": "%(asctime)s", "level": "%(levelname)s", '
                    '"name": "%(name)s", "message": "%(message)s"}'
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": file_level,
                "formatter": "json",
                "filename": log_file,
                "mode": "a",
            },
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            }
        },
    }
