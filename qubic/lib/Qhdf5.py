import json
from typing import Any, Dict

import h5py
import numpy as np


class HDF5Dict:
    """
    Save / load a Python dict to/from an HDF5 file.

    Behavior:
    - numpy arrays -> datasets (with optional compression)
    - homogeneous numeric lists/tuples -> datasets
    - homogeneous string lists/tuples -> datasets (variable-length UTF-8)
    - scalar numbers / bool / short strings -> attributes on the parent group
    - dict -> subgroup: subgroup.attrs['__json__'] stores the full dict as JSON for easy recovery;
             then we also recursively try to store inner items (so both JSON and structured form exist)
    - otherwise (mixed lists, custom objects) -> JSON-serialized into an attribute
    """

    def __init__(self, compression: str | None = "gzip", compression_opts: int | None = 4):
        self.compression = compression
        self.compression_opts = compression_opts

    # --- Public API -----------------------------------------------------
    def save(self, filename: str, data: Dict[str, Any], mode: str = "w") -> None:
        """Write the top-level dict `data` into `filename`."""
        with h5py.File(filename, mode) as h5f:
            self._write_group(h5f, data)

    def load(self, filename: str) -> Dict[str, Any]:
        """Load the HDF5 file and return a Python dict containing original data."""
        with h5py.File(filename, "r") as h5f:
            return self._read_group(h5f)

    # --- Internal helpers -----------------------------------------------
    def _write_group(self, h5group: h5py.Group, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self._write_item(h5group, key, value)

    def _write_item(self, h5group: h5py.Group, name: str, value: Any) -> None:
        # numpy arrays
        if isinstance(value, np.ndarray):
            h5group.create_dataset(name, data=value, compression=self.compression, compression_opts=self.compression_opts)
            return

        # scalars: numbers and bools -> attributes
        if isinstance(value, (int, float, bool)):
            h5group.attrs[name] = value
            return

        # lists / tuples
        if isinstance(value, (list, tuple)):
            # empty -> store as JSON attribute
            if len(value) == 0:
                h5group.attrs[name] = json.dumps(value)
                return

            # homogeneous numeric?
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in value):
                arr = np.asarray(value)
                h5group.create_dataset(name, data=arr, compression=self.compression, compression_opts=self.compression_opts)
                return

            # homogeneous strings?
            if all(isinstance(x, str) for x in value):
                dt = h5py.string_dtype(encoding="utf-8")
                h5group.create_dataset(name, data=np.array(value, dtype=object), dtype=dt, compression=self.compression, compression_opts=self.compression_opts)
                return

            # fallback: serialize to JSON attribute
            try:
                h5group.attrs[name] = json.dumps(value)
            except TypeError:
                h5group.attrs[name] = json.dumps([repr(x) for x in value])
            return

        # dicts -> group (plus a JSON backup)
        if isinstance(value, dict):
            subgroup = h5group.create_group(name)
            try:
                subgroup.attrs["__json__"] = json.dumps(value)
            except TypeError:
                subgroup.attrs["__json__"] = json.dumps(self._safe_dict_for_json(value))

            for k, v in value.items():
                subgroup_key = self._get_key(k)
                self._write_item(subgroup, subgroup_key, v)
            return

        # fallback for arbitrary objects
        try:
            h5group.attrs[name] = json.dumps(value)
        except TypeError:
            h5group.attrs[name] = repr(value)

    def _read_group(self, h5group: h5py.Group) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # first: attributes
        for attr_key, attr_val in h5group.attrs.items():
            if attr_key == "__json__":
                continue
            if isinstance(attr_val, bytes):
                try:
                    attr_val = attr_val.decode("utf-8")
                except Exception:
                    pass
            if isinstance(attr_val, str):
                try:
                    parsed = json.loads(attr_val)
                    out[attr_key] = parsed
                    continue
                except Exception:
                    out[attr_key] = attr_val
                    continue
            out[attr_key] = attr_val

        # second: datasets + subgroups
        for key in h5group:
            obj = h5group[key]
            if isinstance(obj, h5py.Dataset):
                val = obj[()]
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8")
                    except Exception:
                        pass
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val.tolist()
                out[key] = val

            elif isinstance(obj, h5py.Group):
                sub = self._read_group(obj)

                if "__json__" in obj.attrs:
                    try:
                        sub_json = obj.attrs["__json__"]
                        if isinstance(sub_json, bytes):
                            sub_json = sub_json.decode("utf-8")
                        parsed = json.loads(sub_json)

                        # --- FIX: unwrap {"key": {...}} so no double nesting ---
                        if isinstance(parsed, dict) and len(parsed) == 1 and key in parsed:
                            out[key] = parsed[key]
                        else:
                            out[key] = parsed
                        continue
                    except Exception:
                        pass

                out[key] = sub

        return out

    @staticmethod
    def _get_key(key: str) -> str:
        if not isinstance(key, str):
            key = str(key)
        return key.replace("/", "_").strip()

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
