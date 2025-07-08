# Copyright 2025 Dell Inc. or its subsidiaries. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Callable, Dict, List, Union, Tuple
from dqubo import DQUBO
import scipy
import numpy as np
import base64
import zlib
import io


def _serialize_and_encode(
    data: Any, serializer: Callable, compress: bool = True, **kwargs: Any
) -> str:
    """Serializes the input data and return the encoded string.

    Args:
        data: Data to be serialized.
        serializer: Function used to serialize data.
        compress: Whether to compress the serialized data.
        kwargs: Keyword arguments to pass to the serializer.

    Returns:
        String representation.
    """
    buff = io.BytesIO()
    serializer(buff, data, **kwargs)
    buff.seek(0)
    serialized_data = buff.read()
    buff.close()
    if compress:
        serialized_data = zlib.compress(serialized_data)
    return base64.standard_b64encode(serialized_data).decode("utf-8")


def _decode_and_deserialize(
    data: str, deserializer: Callable, decompress: bool = True
) -> Any:
    """Decodes and deserializes input data.

    Args:
        data: Data to be deserialized.
        deserializer: Function used to deserialize data.
        decompress: Whether to decompress.

    Returns:
        Deserialized data.
    """
    buff = io.BytesIO()
    decoded = base64.standard_b64decode(data)
    if decompress:
        decoded = zlib.decompress(decoded)
    buff.write(decoded)
    buff.seek(0)
    orig = deserializer(buff)
    buff.close()
    return orig


class My_RuntimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, DQUBO):
            value = vars(obj)

            return {"__type__": "dqubo", "__value__": value}

        if isinstance(obj, scipy.sparse.spmatrix):
            value = _serialize_and_encode(obj, scipy.sparse.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}

        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return {"__type__": "ndarray", "__value__": obj.tolist()}
            value = _serialize_and_encode(obj, np.save, allow_pickle=False)
            return {"__type__": "ndarray", "__value__": value}

        if isinstance(obj, Tuple):
            return {"__type__": "tuple", "__value__": obj}

        return super().default(obj)


class My_RuntimeDecoder(json.JSONDecoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Any:
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_val = obj["__value__"]

            if obj_type == "dqubo":
                dqubo_obj = DQUBO()

                for prop, value in obj_val.items():
                    setattr(dqubo_obj, prop, value)

                return dqubo_obj

            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, scipy.sparse.load_npz, False)

            if obj_type == "ndarray":
                if isinstance(obj_val, list):
                    return np.array(obj_val)
                return _decode_and_deserialize(obj_val, np.load)

            if obj_type == "tuple":
                return Tuple(obj_val)

        return obj
