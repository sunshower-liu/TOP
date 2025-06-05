# The code for the TOP editing
## Set up

Download the raw models under this folder or customize your own model path.

Create the "*data/stats*" folder.

## Example
You can use edit.py for certain examples of TOP editing.
```bash
python edit.py --alg-name TOPKE --model Qwen2.5-7B-Instruct
```
For evaluation, see tutorial under the *evaluation* folder.

Our code is based on [ROME](https://github.com/kmeng01/rome) and [MEMIT](https://github.com/kmeng01/memit).

## Citation
Xiyu Liu, Zhengxiao Liu, Naibin Gu, Zheng Lin, Ji Xiang, Weiping Wang. Unveiling and Eliminating the Shortcut Learning for Locate-Then-Edit Knowledge Editing via Both Subject and Relation Awareness. arXiv preprint arXiv:2506.04042 (2025)

```text
@misc{liu2025unveilingeliminatingshortcutlearning,
      title={Unveiling and Eliminating the Shortcut Learning for Locate-Then-Edit Knowledge Editing via Both Subject and Relation Awareness}, 
      author={Xiyu Liu and Zhengxiao Liu and Naibin Gu and Zheng Lin and Ji Xiang and Weiping Wang},
      year={2025},
      eprint={2506.04042},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04042}, 
}
```