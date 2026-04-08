# AVA 交互式行为标注工具使用手册

## 1. 这个程序是做什么的

这个程序用于给 AVA 风格数据做交互式行为标注，主要解决这三件事：

- 给每个检测框打 `id`（目标身份）
- 给每个检测框打 `action`（行为类别）
- 通过跟踪预测减少重复手工操作

程序支持“按猪分轮次”标注：先完整标完一头猪的全时间线，再切换下一头猪。

---

## 2. 你需要准备什么

### 2.1 Python 环境

推荐环境：

`D:\Anaconda\envs\ultralytics\python.exe`

### 2.2 依赖

必需：

- `opencv-python`
- `numpy`
- `pandas`

可选：

- `sort-tracker`（若安装失败，程序会自动使用内置 fallback 跟踪器，不影响使用）

### 2.3 数据

你至少需要：

- 图片序列根目录（例如 `rawframes`）
- 原始标注 CSV

支持的原始 CSV 风格（默认）：

`video,frame,x1,y1,x2,y2,action,id`

说明：

- 可无表头
- `id=-1` 表示未标注
- 坐标支持归一化

---

## 3. 安装步骤（给新同事可直接照做）

### 3.1 激活环境

```powershell
conda activate ultralytics
```

### 3.2 安装必需依赖

```powershell
python -m pip install -U opencv-python numpy pandas
```

### 3.3 可选安装官方 SORT

```powershell
python -m pip install -U sort-tracker
```

如果这一步在 Windows 因编译工具失败，可以跳过，程序会用内置 `fallback` 跟踪器。

### 3.4 验证可运行

```powershell
python H:\pycharm_project\trackID\interactive_ava_annotator.py --help
```

---

## 4. 一条命令直接启动

```powershell
& "D:\Anaconda\envs\ultralytics\python.exe" "H:\pycharm_project\trackID\interactive_ava_annotator.py" `
  --image-dir "H:\pycharm_project\PythonProject3\output_data\rawframes" `
  --bbox-csv "H:\pycharm_project\mmaction2\data\annation\fixed_train_sanitized.csv" `
  --output-csv "H:\pycharm_project\trackID\ava_annotations_out.csv" `
  --active-id 1 `
  --round-by-id `
  --auto-next-on-click `
  --recent-memory-frames 4 `
  --autosave-every 100
```

---

## 5. 标注流程（按猪一轮一轮）

1. 按 `i` 设置当前猪的 `ActiveID`
2. 左键点击目标框进行打标
3. 默认会自动下一帧
4. 到最后一帧后，按 `r` 开始下一头猪
5. 全程可按 `s` 手动保存，按 `q` 退出并保存

---

## 6. 快捷键与鼠标（单独速查）

### 6.1 键盘快捷键

| 快捷键 | 作用 |
|---|---|
| `i` | 设置当前 `ActiveID` |
| `c` | 清空 `ActiveID` |
| `a` | 设置当前 `ActiveAction` |
| `x` | 清空 `ActiveAction` |
| `n` / 右箭头 | 下一帧 |
| `p` / 左箭头 | 上一帧 |
| `r` | 下一头猪（回到第 1 帧并可设置新 ID） |
| `z` | 撤回上一步 |
| `s` | 立即保存 |
| `q` | 退出并保存 |

### 6.2 鼠标操作

| 操作 | 作用 |
|---|---|
| 左键点击框 | 用当前 `ActiveID` 赋值给该框 |
| 中键点击框 | 从该框“吸取”ID 作为当前 `ActiveID` |
| 右键点击框 | 删除该框 `id+action` |
| `Ctrl + 右键` 点击框 | 仅删除该框 `id` |
| 右键点击空白区域 | 撤回上一步 |

补充：删除后会锁定该位置，避免预测马上回填。

---

## 7. 保存、续标、共享

### 7.1 保存

- `s` 立刻保存
- `q` 退出时保存
- `--autosave-every N` 每 N 次操作自动保存

### 7.2 下次继续标注

默认开启 `--resume-existing`：

- 如果 `--output-csv` 已存在，程序会自动从该文件加载，继续标注
- 所以请保持同一个输出路径

### 7.3 输出格式

默认输出为无表头：

`video,frame,x1,y1,x2,y2,action,id`

和原始输入风格一致，方便直接发给别人。

---

## 8. 常见问题

### Q1: 为什么看到 `SORT:fallback`？

说明未安装官方 `sort-tracker`，程序正在使用内置跟踪器，这是正常可用状态。

### Q2: 右键删了为什么以前会“又回来”？

新版已修复：右键删除会加禁预测回填逻辑，不会立即被预测写回。

### Q3: ID 交换（IDSW）导致同帧重复怎么办？

新版已在预测阶段和显示阶段都做了同帧唯一约束，并优先保留 `manual > input > predicted`。

### Q4: 如何降低历史污染？

使用 `--recent-memory-frames`，建议从 `4~8` 试起，值越小越偏“近几帧预测”。

---

## 9. 发给别人时建议包含

- `interactive_ava_annotator.py`
- 本说明文档 `AVA_Annotator_使用说明.md`
- 示例启动命令
- 一份示例输入 CSV（可选）
