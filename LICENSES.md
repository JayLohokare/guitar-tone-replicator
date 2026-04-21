# Third-Party Licenses

This project uses the following third-party libraries. All are compatible with commercial use.

## Direct Dependencies

| Package | License | Commercial Use |
|---------|---------|---------------|
| PyTorch | BSD-3-Clause | ✅ |
| librosa | ISC | ✅ |
| soundfile | BSD-3-Clause | ✅ |
| auraloss | Apache-2.0 | ✅ |
| PyTorch Lightning | Apache-2.0 | ✅ |
| torchmetrics | Apache-2.0 | ✅ |
| FastAPI | MIT | ✅ |
| uvicorn | BSD-3-Clause | ✅ |
| Starlette | BSD-3-Clause | ✅ |
| Requests | Apache-2.0 | ✅ |
| pydantic | MIT | ✅ |
| pydub | MIT | ✅ |
| tqdm | MPL-2.0 / MIT | ✅ |
| yt-dlp | Unlicense (public domain) | ✅ |

## Transitive Dependencies

| Package | License | Notes |
|---------|---------|-------|
| numpy | BSD-3-Clause / MIT / Zlib | ✅ All permissive |
| scipy | BSD-3-Clause (main) | ✅ Main library is BSD. Bundled LAPACK/FFTPACK have separate licenses but are not linked derivatively. |
| scikit-learn | BSD-3-Clause | ✅ |
| numba | BSD-2-Clause | ✅ |
| llvmlite | Apache-2.0 + LLVM exception | ✅ |
| soxr | LGPL-2.1-or-later | ✅ Used as dynamic library (pip wheel). LGPL permits commercial use with dynamic linking. Our Python venv is called as subprocess, fully compliant. |
| aiohttp | Apache-2.0 / MIT | ✅ |
| Jinja2 | BSD-3-Clause | ✅ |

## Summary

**No GPL-licensed code is directly used or statically linked.**

All dependencies are permissively licensed (BSD, MIT, ISC, Apache-2.0) or LGPL (used via dynamic linking as subprocess). This project is safe for commercial distribution.

For Mac App Store distribution, the Python runtime runs as a separate subprocess process, maintaining LGPL compliance for `soxr` dynamic linking requirements.

---

## Full License Texts

### BSD-3-Clause (PyTorch, librosa, soundfile, scipy, scikit-learn, numpy, uvicorn, Starlette, Jinja2)

Copyright (c) <year>, <copyright holder>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### MIT (FastAPI, pydantic, pydub, aiohttp partial)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Apache-2.0 (PyTorch Lightning, torchmetrics, auraloss, Requests, llvmlite, aiohttp partial)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### ISC (librosa)

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

### Unlicense (yt-dlp)

This is free and unencumbered software released into the public domain. Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.