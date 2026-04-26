"""Hugging Face Space entrypoint for EIE Economic Intelligence Engine."""

from __future__ import annotations

from demo_gradio import APP_CSS, build_ui


demo = build_ui()


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
