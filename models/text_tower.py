from typing import NamedTuple

import ftfy
import torch
from transformers import AutoTokenizer, T5EncoderModel, GemmaTokenizerFast, T5GemmaModel
from transformers.modeling_utils import ModuleUtilsMixin

import html
import re
import urllib.parse as ul

# Exposing some pre-defined configs
TextTowerPresets = {
    "t5gemma2b-256-bf16": {  # t5gemma
        "model_name": "google/t5gemma-2b-2b-ul2",
        "model_max_length": 256,
        "use_attn_mask": True,
        "use_last_hidden_state": True,
    },
}


class TokenResult(NamedTuple):
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    num_pad_tokens: int | None = None


class TextTower(torch.nn.Module, ModuleUtilsMixin):
    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    # TODO: would probably be easier to define one base text tower class,
    # and then inherit from it to create the different configs

    def __init__(
        self,
        model_name: str = "t5gemma2b-256-bf16",
        only_tokenizer: bool = False,
        use_attn_mask: bool = True,
        model_max_length: int = 256,
        use_last_hidden_state: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        unpadded: bool = False,
    ) -> None:
        super().__init__()
        self.only_tokenizer = only_tokenizer
        self.use_attn_mask = use_attn_mask
        self.use_last_hidden_state = use_last_hidden_state
        self.torch_dtype = torch_dtype
        self.unpadded = unpadded

        self.tokenizer, self.text_encoder = self.create_model(model_name, model_max_length)
        self.tokenizer_max_length = model_max_length
        self.hidden_size = self.text_encoder.config.hidden_size if self.text_encoder else 0
        self.eval()

    @staticmethod
    def basic_clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_text(self, text: str) -> str:
        # See Deepfloyd https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/modules/t5.py
        text = str(text)
        text = ul.unquote_plus(text)
        text = text.strip().lower()
        text = re.sub("<person>", "person", text)

        # Remove all urls:
        text = re.sub(
            r"\b((?:https?|www):(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@))",  # noqa
            "",
            text,
        )  # regex for urls

        # @<nickname>
        text = re.sub(r"@[\w\d]+\b", "", text)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        text = re.sub(r"[\u31c0-\u31ef]+", "", text)
        text = re.sub(r"[\u31f0-\u31ff]+", "", text)
        text = re.sub(r"[\u3200-\u32ff]+", "", text)
        text = re.sub(r"[\u3300-\u33ff]+", "", text)
        text = re.sub(r"[\u3400-\u4dbf]+", "", text)
        text = re.sub(r"[\u4dc0-\u4dff]+", "", text)
        text = re.sub(r"[\u4e00-\u9fff]+", "", text)
        #######################################################

        # все виды тире / all types of dash --> "-"
        text = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            text,
        )

        # кавычки к одному стандарту
        text = re.sub(r"[`´«»“”¨]", '"', text)
        text = re.sub(r"[‘’]", "'", text)

        # &quot;
        text = re.sub(r"&quot;?", "", text)
        # &amp
        text = re.sub(r"&amp", "", text)

        # ip addresses:
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)

        # article ids:
        text = re.sub(r"\d:\d\d\s+$", "", text)

        # \n
        text = re.sub(r"\\n", " ", text)

        # "#123"
        text = re.sub(r"#\d{1,3}\b", "", text)
        # "#12345.."
        text = re.sub(r"#\d{5,}\b", "", text)
        # "123456.."
        text = re.sub(r"\b\d{6,}\b", "", text)
        # filenames:
        text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)

        #
        text = re.sub(r"[\"\']{2,}", r'"', text)  # """AUSVERKAUFT"""
        text = re.sub(r"[\.]{2,}", r" ", text)  # """AUSVERKAUFT"""

        text = re.sub(self.bad_punct_regex, r" ", text)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        text = re.sub(r"\s+\.\s+", r" ", text)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, text)) > 3:
            text = re.sub(regex2, " ", text)

        text = self.basic_clean(text)

        text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)  # jc6640
        text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)  # jc6640vc
        text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)  # 6640vc231

        text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
        text = re.sub(r"(free\s)?download(\sfree)?", "", text)
        text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
        text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
        text = re.sub(r"\bpage\s+\d+\b", "", text)

        text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)  # j2d1a2a...

        text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

        text = re.sub(r"\b\s+\:\s+", r": ", text)
        text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        text.strip()

        text = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", text)
        text = re.sub(r"^[\'\_,\-\:;]", r"", text)
        text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
        text = re.sub(r"^\.\S+$", "", text)

        return text.strip()

    def text_to_token(self, texts: str | list[str]) -> TokenResult:
        """
        returns tokens (B, model_id, SeqLen), attention_mask (B, model_id, SeqLen), num_pad_tokens (optional)
        """
        if isinstance(texts, str):
            texts = [texts]

        # clean text
        texts = [self.clean_text(text) for text in texts]

        if self.unpadded:
            tokens = self.tokenizer(
                texts,
                padding="do_not_pad",
                max_length=self.tokenizer_max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            num_tokens = tokens["input_ids"].shape[1]
            minimum_padding = max(128, num_tokens + 1)
            num_pad_tokens = min(minimum_padding, self.tokenizer_max_length)
            tokens["input_ids"] = torch.nn.functional.pad(
                tokens["input_ids"],
                (0, self.tokenizer_max_length - num_tokens),
                value=self.tokenizer.pad_token_id,
                mode="constant",
            )
            # pad attention mask
            tokens["attention_mask"] = torch.nn.functional.pad(
                tokens["attention_mask"], (0, self.tokenizer_max_length - num_tokens), value=0, mode="constant"
            )

            return TokenResult(tokens["input_ids"], tokens["attention_mask"].bool(), num_pad_tokens)

        # tokenize text
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return TokenResult(tokens["input_ids"], tokens["attention_mask"].bool())

    @torch.no_grad()  # type: ignore
    def token_to_embed(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # Note: /!\ Explicitly disabling autocast here
        with torch.autocast("cuda", enabled=False):
            emb = self.text_encoder(  # type: ignore
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=not self.use_last_hidden_state,
            )
        # this should be 0s: text_encoder.text_model.final_layer_norm(emb_i["hidden_states"][-1]) - emb_i["last_hidden_state"]
        text_embeds = emb["last_hidden_state"] if self.use_last_hidden_state else emb["hidden_states"][-2]

        # return a dict with the text embeddings
        output = {"text_embed": text_embeds}

        return output

    def text_to_embed(self, texts: str | list[str]) -> dict[str, torch.Tensor]:
        token, attention_mask, num_pad_tokens = self.text_to_token(texts)
        token, attention_mask = token.to(self.device), attention_mask.to(self.device)
        text_encoder_out = self.token_to_embed(token, attention_mask)

        if self.unpadded and num_pad_tokens is not None:
            # unpad text embeddings to the number of tokens in the text (minimum 128)
            text_encoder_out["text_embed"] = text_encoder_out["text_embed"][:, :num_pad_tokens]

        if self.use_attn_mask:
            # Keep only the mask corresponding to the indices of max sums along the 'model_id' dimension
            text_encoder_out["attention_mask"] = attention_mask
            # unpad attention mask to the number of tokens in the text (minimum 128)
            if self.unpadded and num_pad_tokens is not None:
                text_encoder_out["attention_mask"] = text_encoder_out["attention_mask"][:, :num_pad_tokens]

        return text_encoder_out  # type: ignore

    def forward(self, texts: str | list[str]) -> dict[str, torch.Tensor]:
        return self.text_to_embed(texts)

    def create_model(self, model_config: str, model_max_length: int) -> tuple[AutoTokenizer, T5EncoderModel | None]:
        tokenizer = GemmaTokenizerFast.from_pretrained(model_config)
        tokenizer.model_max_length = model_max_length
        if self.only_tokenizer:
            return tokenizer, None
        else:
            text_encoder = T5GemmaModel.from_pretrained(model_config, torch_dtype=self.torch_dtype).encoder
        return tokenizer, text_encoder