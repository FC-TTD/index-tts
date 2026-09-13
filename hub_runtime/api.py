"""Register the established API without embedding lifecycle policy."""

from functools import partial
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from .service import synthesize


def build_api(args, runtime):
    app = FastAPI(
        title="IndexTTS API",
        description="IndexTTS2.5 语音合成 API 服务",
        version="2.5.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    endpoint = partial(
        runtime.task(synthesize) if runtime is not None else synthesize, runtime, args
    )
    endpoint.__name__ = "generate_audio"
    app.add_api_route(
        "/generate",
        endpoint,
        methods=["POST"],
        response_class=Response,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                }
            }
        },
    )
    return app
