#!/usr/bin/env bash
# 三测第 1 步：按冻结镜像在云侧/边侧宿主机上创建 third-test 容器。
#
# 部署位置：/data1/tangruijing/third-test/create_docker.sh（云侧 powerone1 宿主机）
# 用法：
#   bash /data1/tangruijing/third-test/create_docker.sh
#   FORCE=1 bash .../create_docker.sh     # 容器里还有训练进程时也强行重建
#
# 做完这一步，第 2 步就是：
#   python3 /data1/tangruijing/third-test/run_docker.py
# 它会自己跑 polar 和 megatron baseline 各 20 iter 并出对比。
#
# 本脚本只创建容器 + 自检，不跑训练。
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_HOST="${EDGE_HOST:-powerone2}"
CONTAINER="${CONTAINER:-third-test}"
CLOUD_IMAGE="${CLOUD_IMAGE:-udtca-third-test:cloud-20261105}"
EDGE_IMAGE="${EDGE_IMAGE:-udtca-third-test:edge-20261105}"
DEPLOY_DIR_REMOTE="${DEPLOY_DIR_REMOTE:-/data1/tangruijing/third-test}"
FORCE="${FORCE:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

step() { printf '\n=== %s ===\n' "$*"; }
die()  { printf '失败: %s\n' "$*" >&2; exit 1; }
on_edge() { ssh "${SSH_OPTS[@]}" "$EDGE_HOST" "$@"; }

step "0. 位置确认（必须在云侧宿主机跑）"
echo "宿主机: $(hostname)  部署目录: $DEPLOY_DIR  边侧: $EDGE_HOST"
[ -f "$DEPLOY_DIR/docker/run_cloud_container.sh" ] || die "缺 $DEPLOY_DIR/docker/run_cloud_container.sh"
on_edge "test -f $DEPLOY_DIR_REMOTE/docker/run_edge_container.sh" \
  || die "边侧缺 $DEPLOY_DIR_REMOTE/docker/run_edge_container.sh"

step "1. 镜像检查"
docker image inspect "$CLOUD_IMAGE" >/dev/null 2>&1 || die "本机缺镜像 $CLOUD_IMAGE"
on_edge "docker image inspect $EDGE_IMAGE >/dev/null" 2>/dev/null || die "边侧缺镜像 $EDGE_IMAGE"
echo "云侧 $CLOUD_IMAGE : $(docker image inspect -f '{{.Id}}' "$CLOUD_IMAGE")"
echo "边侧 $EDGE_IMAGE  : $(on_edge "docker image inspect -f '{{.Id}}' $EDGE_IMAGE")"

step "2. 确认没有在跑的训练（有就停下，除非 FORCE=1）"
busy_cloud=$(docker exec "$CONTAINER" pgrep -f 'run_qwen14b|torchrun|megatron_probe' 2>/dev/null | head -1 || true)
busy_edge=$(on_edge "docker exec $CONTAINER pgrep -f 'run_qwen14b|torchrun|megatron_probe' 2>/dev/null | head -1" || true)
if [ -n "$busy_cloud$busy_edge" ] && [ "$FORCE" != "1" ]; then
  die "容器里还有训练进程（云:${busy_cloud:-无} 边:${busy_edge:-无}），确认没在用后 FORCE=1 重跑"
fi

step "3. 边侧创建容器"
on_edge "bash $DEPLOY_DIR_REMOTE/docker/run_edge_container.sh"

step "4. 云侧创建容器"
bash "$DEPLOY_DIR/docker/run_cloud_container.sh"

step "5. 自检：容器里的代码/语料/依赖全部来自镜像"
CHECK_PY="$(mktemp)"
trap 'rm -f "$CHECK_PY"' EXIT
cat > "$CHECK_PY" <<'PY'
import os, glob, subprocess, sys

ok = True
def bad(msg):
    global ok
    ok = False
    print("    [FAIL] " + msg)

import torch
print("  python %s / torch %s" % (sys.version.split()[0], torch.__version__))

# 项目代码必须来自镜像里的 /workspace；其余第三方包在 site-packages（即环境）
for name in ("psgd", "bitscom", "tensorboard", "datasets"):
    mod = __import__(name)
    f = getattr(mod, "__file__", "?")
    print("  %-11s %s" % (name, f))
    if name in ("psgd", "bitscom") and not f.startswith("/workspace/"):
        bad("%s 不在 /workspace 下" % name)

try:
    from bitscom._lowbit_c import ProcessGroupLowBit  # noqa: F401
    print("  bitscom C 扩展加载 OK")
except Exception as e:
    bad("bitscom C 扩展加载失败: %r" % (e,))

data = "/workspace/megatron-data/fineweb_sample-10BT"
for pat in ("*.bin", "*.idx"):
    got = glob.glob(os.path.join(data, pat))
    print("  %s %s" % (pat, got or "缺失"))
    if not got:
        bad("megatron 语料缺 %s" % pat)

polar_cache = glob.glob(
    "/workspace/udtca/experiments/qwen14b/cache/*_seq256_samples*_labels_unshifted_v2.pt")
print("  polar token 缓存 %s" % (polar_cache or "缺失"))
if not polar_cache:
    bad("polar 数据集缓存缺失（离线模式连不上 HF，跑不起来）")

snap = "/workspace/hf-cache/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots"
print("  tokenizer 快照 %s" % ("在位" if os.path.isdir(snap) else "缺失"))
if not os.path.isdir(snap):
    bad("tokenizer 缓存缺失")
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    print("  tokenizer 离线加载 OK, vocab=%d" % tok.vocab_size)

sp = "/usr/local/lib/python3.12/site-packages"
for name in ("__editable__.psgd-0.1.0.pth", "__editable__.bitscom-0.1.0.pth"):
    p = os.path.join(sp, name)
    if not os.path.exists(p):
        bad("缺 %s" % p)
        continue
    with open(p) as fh:
        targets = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    print("  %-32s -> %s" % (name, targets))
    for t in targets:
        if not t.startswith("/workspace/"):
            bad("%s 指向 /workspace 之外: %s" % (name, t))

r = subprocess.run(["md5sum", "-c", "--quiet", "/workspace/udtca_manifest.txt"],
                   cwd="/workspace/udtca", capture_output=True, text=True)
with open("/workspace/udtca_manifest.txt") as fh:
    lines = sum(1 for _ in fh)
print("  md5 清单校验 %d 个文件: %s" % (lines, "OK" if r.returncode == 0 else "FAILED"))
if r.returncode != 0:
    bad("代码清单校验失败: " + (r.stdout + r.stderr)[:500])

# /workspace 里不能有指向镜像外的符号链接
outside = []
for root, dirs, files in os.walk("/workspace"):
    for n in dirs + files:
        p = os.path.join(root, n)
        if os.path.islink(p):
            tgt = os.path.realpath(p)
            if not (tgt.startswith("/workspace/") or tgt == "/workspace"):
                outside.append("%s -> %s" % (p, tgt))
print("  /workspace 外部符号链接: %s" % (outside[:5] if outside else "无"))
if outside:
    bad("/workspace 里有 %d 个链接指向镜像外" % len(outside))

sys.exit(0 if ok else 1)

PY

echo "-- 云侧 --"
docker exec -i "$CONTAINER" python - < "$CHECK_PY" || die "云侧自检失败"
echo "-- 边侧 --"
on_edge "docker exec -i $CONTAINER python -" < "$CHECK_PY" || die "边侧自检失败"

step "完成"
cat <<MSG
容器已就绪（云侧/边侧各一台，名字 $CONTAINER）。下一步跑训练：

  python3 $DEPLOY_DIR/run_docker.py                 # polar + megatron 各 20 iter，自动对比
  python3 $DEPLOY_DIR/run_docker.py --preflight-only  # 只想先查卡/连通性
MSG
