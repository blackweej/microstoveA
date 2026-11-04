"""make가 없는 곳에서 Makefile과 같은 빌드를 돌린다.

    python tools/build.py lib      foodcv.dll (Windows) / libfoodcv.so (Linux)
    python tools/build.py test     C 단위 테스트
    python tools/build.py debug    -O0, UBSan으로 C 단위 테스트

컴파일러는 환경변수 CC, gcc, `python -m ziglang cc` 순서로 고른다.
"""
import glob, os, shlex, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
FLAGS = ["-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-Iinclude"]


def compiler():
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"])
    if shutil.which("gcc"):
        return ["gcc"]
    return [sys.executable, "-m", "ziglang", "cc"] + (["-target", "x86_64-windows-gnu"] if WIN else [])


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "lib"
    src = sorted(glob.glob("src/*.c", root_dir=ROOT))
    tests = sorted(glob.glob("tests/*.c", root_dir=ROOT))
    exe = "run_tests.exe" if WIN else "run_tests"
    cc = compiler()
    if target == "lib":
        out = "foodcv.dll" if WIN else "libfoodcv.so"
        pic = [] if WIN else ["-fPIC"]
        return subprocess.call(cc + FLAGS + ["-O3", "-shared"] + pic + ["-o", out] + src + ["-lm"], cwd=ROOT)
    if target in ("test", "debug"):
        opt = ["-O3"] if target == "test" else ["-O0", "-g", "-fsanitize=undefined"]
        rc = subprocess.call(cc + FLAGS + opt + ["-DFCV_DEBUG_DUMP", "-o", exe] + tests + src + ["-lm"], cwd=ROOT)
        return rc or subprocess.call([os.path.join(ROOT, exe)], cwd=ROOT)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
