## 安装

需要的环境在 ``requirements.txt``。需要注意的是，当安装完成后，需要降低``Werkzeug``的版本
``pip install Werkzeug==2.2.2``

## 配置

``configs/config.custom.yaml`` 里面配置文件。

需要配置``openai api_key``、``huggingface token``、``device``。

使用的OpenAI模型在``get_token_ids.py``里面配置。目前支持的模型有

```markdown
gpt-4
gpt-4-1106-preview
gpt-3.5-turbo
gpt-3.5-turbo-1106
```

建议及时查阅OpenAI [官方文档](https://platform.openai.com/docs/models) 以了解最新模型情况。

如果需要加入新的模型，请修改``get_token_ids.py``中的相关内容。

## 下载本地模型

当前版本支持 image-captioning 的一个本地模型（仅用作示例）。下载这些模型请使用

```bash
cd models
./download.sh
```

小模型将会被自动下载到 `models` 目录下（请注意不要改动模型下载地址）


### 加入新的本地小模型

如果要加入新的本地小模型，请使用以下步骤。

1. 将模型加入 `data/p0_models.jsonl` （请参考里面已有模型的样式）。其中`id`是模型的路劲（在`models`目录下的相对路径）；并且加入其他对模型的描述（特别是任务类型相关的内容），帮助GPT判断该使用什么模型处理任务。
2. 查看`configs/config.custom.yaml`文件里面的 `tprompt: parse_task` 参数，确保新模型针对的任务类型（例如`text-classification`）在 Prompt 的任务类型列表里面。同时，确定prompt的`"args"`下面是否包括了相应参数。（总之要仔细检查一下prompt，确保GPT能够提取出相应的输入信息）
3. 将模型按照 `id` 所述路径，放到 `models` 目录下。
4. 在 `server.py` 中，在`load_pipes`方法，以及`models`方法中，加入模型处理方式。

## 运行

分为两步

### 运行本地模型

首先需要运行本地的模型服务。需要把``[path to the config file]`` 替换为实际的配置文件。

```python
python server.py --config [path to the config file]
```

该命令会启动``models``里面的模型。

### 运行服务

执行下面的命令运行程序， ``[path to the config file]`` 需要与上面的配置文件一致。

```python
python chat.py --config [path to the config file]
```

之后可以交互。

注意：如果输入包括文件，请一定把文件放在 `public/example` 目录下，并且给出需要处理的文件的文件路径。


