from fastapi import APIRouter

from pydantic import BaseModel

from app.services.repo_service import (
clone_repository
)

router=APIRouter()


class RepoInput(
BaseModel
):
    repo_url:str


@router.post(
"/ingest"
)

def ingest(
data:RepoInput
):

    path=clone_repository(
        data.repo_url
    )

    return {
        "stored":path
    }