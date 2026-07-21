"""HTTP request schemas."""

from __future__ import annotations

from datetime import date
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from model_training import DEFAULT_MODEL_TRAINING_METHOD

from .tagging import normalize_tags


class ExpressionRequest(BaseModel):
    expression: str


class MethodsRequest(BaseModel):
    names: list[str] | None = None


class FactorInput(BaseModel):
    factor_name: str
    batch_id: str | None = None
    expression: str | None = None

    @model_validator(mode="after")
    def reference_or_expression(self) -> "FactorInput":
        if self.expression is None and self.batch_id is None:
            raise ValueError("batch_id is required for a registered factor")
        return self


class GateCondition(BaseModel):
    """One threshold rule on a produced metric (e.g. metric ≥ value)."""

    metric: str = Field(min_length=1, max_length=120)
    op: Literal["gte", "lte", "gt", "lt", "between"] = "gte"
    value: float
    value2: float | None = None

    @model_validator(mode="after")
    def between_needs_upper_bound(self) -> "GateCondition":
        if self.op == "between" and self.value2 is None:
            raise ValueError("区间条件需要提供上下界两个数值")
        return self


class GateSpec(BaseModel):
    """A screening gate: several conditions combined with all/any."""

    conditions: list[GateCondition] = Field(default_factory=list, max_length=20)
    match: Literal["all", "any"] = "all"


class JobCreate(BaseModel):
    kind: Literal["evaluate", "funnel"] = "evaluate"
    title: str | None = None
    factors: list[FactorInput] = Field(default_factory=list, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)
    tag_match: Literal["any", "all"] = "any"
    library: Literal["test", "factor"] = "test"
    methods: list[str] | None = None
    template_id: int | None = None
    horizon: int = Field(default=1, ge=1)
    n_quantiles: int = Field(default=10, ge=2)
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    gate: GateSpec | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        return normalize_tags(values)

    @model_validator(mode="after")
    def factor_or_tag_source(self) -> "JobCreate":
        if self.factors and self.tags:
            raise ValueError("请使用因子列表或标签之一提交任务，不能同时使用")
        if not self.factors and not self.tags:
            raise ValueError("请至少选择一个因子或标签")
        return self


class FactorCreate(BaseModel):
    factor_name: str
    expression: str
    project: str = Field(default="自定义因子", min_length=1, max_length=80)
    paper_expression: str = ""
    uses_proxy: bool = False
    proxy_description: str = ""
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: list[str] | None) -> list[str] | None:
        return normalize_tags(values) if values is not None else None

    @field_validator("project")
    @classmethod
    def validate_project(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("项目不能为空")
        return normalized


class FactorTagsUpdate(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        return normalize_tags(values)


class FactorProjectUpdate(BaseModel):
    project: str = Field(min_length=1, max_length=80)

    @field_validator("project")
    @classmethod
    def validate_project(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("项目不能为空")
        return normalized


class GeneticCampaignCreate(BaseModel):
    """Safe, bounded web input for one resumable GP mining campaign."""

    campaign: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon: int = Field(default=1, ge=1, le=60)
    n_quantiles: int = Field(default=10, ge=3, le=20)
    preprocess_mode: Literal["paper_local", "market_cap", "none"] = "paper_local"
    population_size: int = Field(default=1000, ge=2, le=10000)
    generations: int = Field(default=3, ge=1, le=100)
    hall_of_fame: int = Field(default=100, ge=1, le=5000)
    components: int = Field(default=10, ge=1, le=500)
    tournament_size: int = Field(default=20, ge=1, le=1000)
    n_jobs: int = Field(default=2, ge=1, le=32)
    compute_backend: Literal["cpu", "mps"] = "cpu"
    seed: int = Field(default=20190610, ge=0, le=2**32 - 1)
    continuous: bool = False
    pause_seconds: float = Field(default=60.0, ge=0, le=86400)
    max_cycles: int | None = Field(default=None, ge=1, le=100000)
    admit: bool = True

    @field_validator("campaign")
    @classmethod
    def normalize_campaign(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("train_start", "train_end", "test_start", "test_end")
    @classmethod
    def campaign_iso_date(cls, value: str) -> str:
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("日期必须是 YYYY-MM-DD") from exc

    @model_validator(mode="after")
    def validate_campaign_contract(self) -> "GeneticCampaignCreate":
        if date.fromisoformat(self.train_start) > date.fromisoformat(self.train_end):
            raise ValueError("训练集开始日期不能晚于结束日期")
        if date.fromisoformat(self.test_start) > date.fromisoformat(self.test_end):
            raise ValueError("测试集开始日期不能晚于结束日期")
        if date.fromisoformat(self.train_end) >= date.fromisoformat(self.test_start):
            raise ValueError("训练集必须严格早于测试集，两个区间不能重叠")
        if self.tournament_size > self.population_size:
            raise ValueError("锦标赛规模不能超过种群规模")
        if self.hall_of_fame > self.population_size * self.generations:
            raise ValueError("Hall of Fame 不能超过所有代的程序总数")
        if self.components > self.hall_of_fame:
            raise ValueError("进入测试集的候选数不能超过 Hall of Fame")
        if self.compute_backend == "mps" and self.n_jobs != 1:
            raise ValueError("MPS 后端需要并行线程设为 1；并行由 GPU 提供")
        if not self.continuous and self.max_cycles is not None:
            raise ValueError("只有连续模式可以设置最大 cycle 数")
        return self


class TemplateInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["methods", "funnel"] = "methods"
    methods: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=1, max_length=20)


class LinearModelTerm(BaseModel):
    """One frozen factor-library component in a linear multi-factor model."""

    batch_id: str = Field(min_length=1, max_length=120)
    factor_name: str = Field(min_length=1, max_length=120)
    weight: float

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("线性权重必须是有限数值")
        return value


class ModelTestInput(BaseModel):
    """Editable configuration for a train-then-holdout model-test session."""

    model_name: str = Field(min_length=1, max_length=120)
    terms: list[LinearModelTerm] = Field(min_length=1, max_length=20)
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon: int = Field(default=1, ge=1, le=60)
    n_quantiles: int = Field(default=10, ge=2, le=20)
    methods: list[str] = Field(min_length=1, max_length=40)
    training_method: str = Field(
        default=DEFAULT_MODEL_TRAINING_METHOD,
        min_length=1,
        max_length=120,
    )
    training_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def strip_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("模型名称不能为空")
        return normalized

    @field_validator("train_start", "train_end", "test_start", "test_end")
    @classmethod
    def iso_date(cls, value: str) -> str:
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("日期必须是 YYYY-MM-DD") from exc

    @field_validator("methods")
    @classmethod
    def nonempty_method_names(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("评价方法名称不能为空")
        return normalized

    @field_validator("training_method")
    @classmethod
    def nonempty_training_method(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("训练方法名称不能为空")
        return normalized

    @model_validator(mode="after")
    def valid_split_and_terms(self) -> "ModelTestInput":
        train_start = date.fromisoformat(self.train_start)
        train_end = date.fromisoformat(self.train_end)
        test_start = date.fromisoformat(self.test_start)
        test_end = date.fromisoformat(self.test_end)
        if train_start > train_end:
            raise ValueError("训练集开始日期不能晚于结束日期")
        if test_start > test_end:
            raise ValueError("测试集开始日期不能晚于结束日期")
        if train_end >= test_start:
            raise ValueError("训练集必须严格早于测试集，两个区间不能重叠")
        identities = [(term.batch_id, term.factor_name) for term in self.terms]
        if len(identities) != len(set(identities)):
            raise ValueError("同一因子只能在线性模型中出现一次")
        if not any(term.weight != 0 for term in self.terms):
            raise ValueError("至少需要一个非零线性权重")
        return self
