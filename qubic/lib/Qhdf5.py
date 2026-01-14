import json
from typing import Any, Dict

import h5py
import numpy as np


class HDF5Dict:
    def __init__(self, compression: str | None = "gzip", compression_opts: int | None = 4):
        self.compression = compression
        self.compression_opts = compression_opts

    # Utilities functions
    def save_dict(self, filename: str, data: Dict[str, Any], mode: str = "w") -> None:
        with h5py.File(filename, mode) as h5f:
            self._write_group(h5f, data)

    def save_array(self, filename: str, data: np.ndarray, mode: str = "w") -> None:
        with h5py.File(filename, mode) as h5f:
            self._write_item(h5f, "data", data)

    def load_dict(self, filename: str) -> Dict[str, Any]:
        with h5py.File(filename, "r") as h5f:
            return self._read_group(h5f)

    def load_array(self, filename: str) -> np.ndarray:
        with h5py.File(filename, "r") as h5f:
            obj = h5f["data"]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError("The key 'data' does not correspond to a dataset.")
            return self._decode_dataset(obj)

    def load_item(self, filename: str, path: str) -> Any:
        if "::" in path:
            h5_path, attr_name = path.split("::", 1)
            with h5py.File(filename, "r") as h5f:
                return self._decode_attribute(h5f[h5_path].attrs[attr_name])

        with h5py.File(filename, "r") as h5f:
            obj = h5f[path]
            if isinstance(obj, h5py.Dataset):
                return self._decode_dataset(obj)
            if isinstance(obj, h5py.Group):
                if "__json__" in obj.attrs:
                    try:
                        raw = obj.attrs["__json__"]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        return json.loads(raw)
                    except Exception:
                        pass
                return self._read_group(obj)

        raise ValueError(f"Unknown object type at path: {path}")

    # Internal helpers
    def _write_group(self, h5group: h5py.Group, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self._write_item(h5group, key, value)

    def _write_item(self, h5group: h5py.Group, name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            h5group.create_dataset(name, data=value, compression=self.compression, compression_opts=self.compression_opts)
            return

        if isinstance(value, (int, float, bool)):
            h5group.attrs[name] = value
            return

        if isinstance(value, (list, tuple)):
            if all(isinstance(x, np.ndarray) for x in value):
                subgroup = h5group.create_group(name)
                subgroup.attrs["__kind__"] = "sequence"
                subgroup.attrs["__sequence_type__"] = "tuple" if isinstance(value, tuple) else "list"
                subgroup.attrs["__length__"] = len(value)

                for i, arr in enumerate(value):
                    if isinstance(arr, np.ndarray) and arr.shape == ():
                        # scalar dataset → NO compression allowed
                        subgroup.create_dataset(str(i), data=arr)
                    else:
                        subgroup.create_dataset(
                            str(i),
                            data=arr,
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                        )

                return

            if len(value) == 0:
                h5group.attrs[name] = json.dumps(value)
                return

            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in value):
                arr = np.asarray(value)
                h5group.create_dataset(name, data=arr, compression=self.compression, compression_opts=self.compression_opts)
                return

            if all(isinstance(x, str) for x in value):
                dt = h5py.string_dtype(encoding="utf-8")
                h5group.create_dataset(name, data=np.array(value, dtype=object), dtype=dt, compression=self.compression, compression_opts=self.compression_opts)
                return

            h5group.attrs[name] = json.dumps(value, default=repr)
            return

        if isinstance(value, dict):
            subgroup = h5group.create_group(name)
            try:
                subgroup.attrs["__json__"] = json.dumps(value)
            except TypeError:
                subgroup.attrs["__json__"] = json.dumps(self._safe_dict_for_json(value))

            for k, v in value.items():
                self._write_item(subgroup, self._get_key(k), v)
            return

        h5group.attrs[name] = json.dumps(value, default=repr)

    def _read_group(self, h5group: h5py.Group) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for k, v in h5group.attrs.items():
            if k == "__json__":
                continue
            out[k] = self._decode_attribute(v)

        for key in h5group:
            obj = h5group[key]

            if isinstance(obj, h5py.Dataset):
                out[key] = self._decode_dataset(obj)
                continue

            if isinstance(obj, h5py.Group):
                if obj.attrs.get("__kind__") == "sequence":
                    length = obj.attrs["__length__"]
                    seq = []
                    for i in range(length):
                        val = obj[str(i)][()]
                        if isinstance(val, np.ndarray) and val.shape == ():
                            val = val.tolist()
                        seq.append(val)

                    out[key] = tuple(seq) if obj.attrs["__sequence_type__"] == "tuple" else seq
                    continue

                if "__json__" in obj.attrs:
                    try:
                        raw = obj.attrs["__json__"]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        out[key] = json.loads(raw)
                        continue
                    except Exception:
                        pass

                out[key] = self._read_group(obj)

        return out

    def _decode_attribute(self, val):
        if isinstance(val, bytes):
            try:
                val = val.decode("utf-8")
            except Exception:
                pass
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return val
        return val

    def _decode_dataset(self, ds):
        val = ds[()]
        if isinstance(val, bytes):
            try:
                val = val.decode("utf-8")
            except Exception:
                pass
        if isinstance(val, np.ndarray) and val.shape == ():
            return val.tolist()
        return val

    @staticmethod
    def _get_key(key: str) -> str:
        return str(key).replace("/", "_").strip()

    @staticmethod
    def _safe_dict_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            try:
                json.dumps({k: v})
                out[k] = v
            except TypeError:
                if isinstance(v, (np.integer, np.floating)):
                    out[k] = v.item()
                else:
                    out[k] = repr(v)
        return out
