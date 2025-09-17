from enum import Enum


class BaseStrEnum(str, Enum):
    @classmethod
    def values(cls):
        return [item.value for item in cls]

    @classmethod
    def names(cls):
        return [item.name for item in cls]

    @classmethod
    def dict(cls):
        return {item.name: item.value for item in cls}


class AllowImageFileExtensions(BaseStrEnum):
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
