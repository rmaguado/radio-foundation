from ..utils import Database


class CtDatabase(Database):
    def __init__(self, config, storage: str = "dicom") -> None:
        super().__init__(config)

        self.storage = storage
        self.modality = "CT"

    def store_metadata(self) -> None:
        self.cursor.execute("SELECT value FROM metadata WHERE key = 'modality';")
        modality_value = self.cursor.fetchone()
        if modality_value and modality_value[0] != "CT":
            raise ValueError(
                f"Existing database modality value is different: {modality_value[0]}"
            )

        self.cursor.execute("SELECT value FROM metadata WHERE key = 'storage';")
        storage_value = self.cursor.fetchone()
        if storage_value and storage_value[0] != self.storage:
            raise ValueError(
                f"Existing database storage value is different: {storage_value[0]}"
            )

        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('modality', ?);",
            (self.modality,),
        )
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('storage', ?);",
            (self.storage,),
        )
