from pathlib import Path

from yubaoBtrader.trader.app import BaseApp
from .base import APP_NAME, StopOrder
from .engine import CtaEngine
from .template import CtaTemplate, CtaSignal, TargetPosTemplate
from .tools import TradeResult, ResultManager, OrderRecorder, PnlTracker


class CtaStrategyApp(BaseApp):
    """"""

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "CTA Strategy"
    engine_class: CtaEngine = CtaEngine
    widget_name: str = "CtaManager"
    icon_name: str = str(app_path.joinpath("ui", "cta.ico"))
