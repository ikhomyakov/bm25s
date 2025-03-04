import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tqdm
import transformers

RANDOM_SEED = 100
DEVICE = "cpu"
EMBEDDING_SIZE = 768
MAX_TEXT_LENGTH = 2048
LANGUAGE_MODEL_NAME = "bert-base-cased"
K1 = 1.2
B = 0.75
K = 10


class LSH:
    """Implements locality sensitive hashing (cosine distance)
    Derived from: MingYu (Ethen) Liu (@ethen8181), Locality
    Sensitive Hashing (LSH) - Cosine Distance,
     http://ethen8181.github.io/machine-learning/recsys/content_based/lsh_text.html
    """

    def __init__(self, dim, n_vectors: int = 16) -> None:
        self.random_vectors = np.random.randn(dim, n_vectors)
        self.powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    def hash(self, a: np.ndarray) -> np.ndarray:
        return (a.dot(self.random_vectors) >= 0).dot(self.powers_of_two)


class BM25S:
    def __init__(
        self,
        language_model,
        tokenizer,
        db_path: str,
        k1: float = 1.2,
        b: float = 0.75,
        s: float = 0.5,
        embedding_size: int = 768,
        max_text_length: int = 512,
        device: str = "cpu",
    ):
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.k1 = k1
        self.b = b
        self.s = s
        self.embedding_size = embedding_size
        self.max_text_length = max_text_length
        self.device = device
        self.lsh = LSH(self.embedding_size)

    def tokenize(self, text: str, padding: bool = False, truncation: bool = False):
        return self.tokenizer(
            [text],
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=self.max_text_length,
        ).to(self.device)

    def create_tables(self, conn) -> None:
        conn.execute("create table m(n, avgdl, k1, b, s)")
        conn.execute(
            "create table d(did primary key, text, dl)"
        )  # did > 0: pos docs, did < 0: neg docs
        conn.execute("create table q(did primary key, text, pos_did, neg_did)")
        conn.execute(
            "create table qt(tid primary key, token, nw)"
        )  # tid > 0: token domain, tid < 0: semantic domain
        conn.execute(
            "create table qtf(did references q(did), tid, tf, primary key (did, tid))"
        )  # tid > 0: token domain, tid < 0: semantic domain
        conn.execute(
            "create table t(tid primary key, token, nw)"
        )  # tid > 0: token domain, tid < 0: semantic domain
        conn.execute(
            "create table tf(tid, did references d(did), tf, primary key (tid, did))"
        )

    def upsert_t(self, conn, inputs, table_name: str = "t") -> None:
        if isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            tids = inputs["input_ids"][0][1:-1].tolist()
            tokens = self.tokens_to_text(inputs)
        elif isinstance(inputs, list):
            tids = inputs
            tokens = [str(x) for x in inputs]
        else:
            raise Exception(f"upsert_t: Bad inputs {inputs=}, {type(inputs)=}")

        assert len(tids) == len(tokens)

        for tid, token in Counter(zip(tids, tokens)):
            conn.execute(
                f"insert into {table_name}(tid, token, nw) values(?, ?, ?)"
                " on conflict(tid) do update set nw = nw + 1",
                (tid, token, 1),
            )

    def insert_tf(self, conn, did, inputs, table_name: str = "tf") -> None:
        if isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            tids = inputs["input_ids"][0][1:-1].tolist()
        elif isinstance(inputs, list):
            tids = inputs
        else:
            raise Exception(f"insert_tf: Bad inputs {inputs=}, {type(inputs)=}")

        for tid, tf in Counter(tids).items():
            conn.execute(
                f"insert into {table_name}(tid, did, tf) values(?, ?, ?)",
                (tid, did, tf),
            )

    def upsert_meta(self, conn) -> None:
        conn.execute("delete from m")
        conn.execute(
            "insert into m(n, avgdl, k1, b, s) select count(*), avg(dl), ?, ?, ? from d",
            (self.k1, self.b, self.s),
        )

    def insert_q(self, conn, qid, q, pos_did, neg_did) -> None:
        conn.execute(
            "insert into q(did, text, pos_did, neg_did) values(?, ?, ?, ?)",
            (qid, q, pos_did, neg_did),
        )

    def insert_d(self, conn, did, d, dt) -> None:
        conn.execute(
            "insert into d(did, text, dl) values(?, ?, ?)",
            (did, d, len(dt["input_ids"][0][1:-1])),
        )

    def tokens_to_text(self, inputs):
        return self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][1:-1])

    def infer(self, text: str):
        inputs = self.tokenize(text, padding=True, truncation=True)
        try:
            outputs = self.language_model(**inputs)
            v = outputs.last_hidden_state.detach()
            outputs.pooler_output.detach()
        except RuntimeError as e:
            logging.error(f"Inference error {text=}, {e=}")
            v = None
        return v[0]

    def semantic_tokenize(self, text: str) -> list[int]:
        v = self.infer(text)[1:-1]
        s = (-self.lsh.hash(v.numpy())).tolist()
        return s

    def ingest_corpus(self, tsv_path: str, start_line_id: int) -> int:
        with sqlite3.connect(self.db_path) as conn:
            self.create_tables(conn)
            with open(tsv_path, "r", encoding="utf_8") as f:
                line_id = start_line_id
                for text in tqdm.tqdm(f):
                    with conn:
                        q, dp, dn = text.strip().split("\t")

                        qs = self.semantic_tokenize(q)
                        dps = self.semantic_tokenize(dp)
                        dns = self.semantic_tokenize(dn)

                        qt = self.tokenize(q)
                        dpt = self.tokenize(dp)
                        dnt = self.tokenize(dn)

                        qid = line_id
                        pos_did = qid
                        neg_did = -qid

                        self.insert_q(conn, qid, q, pos_did, neg_did)
                        self.insert_d(conn, pos_did, dp, dpt)
                        self.insert_d(conn, neg_did, dn, dnt)

                        self.upsert_t(conn, qt, "qt")
                        self.upsert_t(conn, dpt, "t")
                        self.upsert_t(conn, dnt, "t")

                        self.insert_tf(conn, qid, qt, "qtf")
                        self.insert_tf(conn, pos_did, dpt, "tf")
                        self.insert_tf(conn, neg_did, dnt, "tf")

                        self.upsert_t(conn, qs, "qt")
                        self.upsert_t(conn, dps, "t")
                        self.upsert_t(conn, dns, "t")

                        self.insert_tf(conn, qid, qs, "qtf")
                        self.insert_tf(conn, pos_did, dps, "tf")
                        self.insert_tf(conn, neg_did, dns, "tf")

                        line_id += 1

            self.upsert_meta(conn)
        return line_id - start_line_id

    def retrieve_query(self, qid: int) -> list:
        with sqlite3.connect(self.db_path) as conn:
            with conn:
                c = conn.cursor()
                c.execute("select * from q where did = ?", (qid,))
                ds = c.fetchall()
                c.close()
        return detuple(ds)[0]

    def bm25s_by_qid(self, qid: int, k: int) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            with conn:
                c = conn.cursor()
                c.execute(
                    """
                    with b as (
                       select tf.did, tf.tid,
                           2.0 * case sign(tf.tid) when 1 then 1.0 - m.s else m.s end
                           * tf.tf * (1 + m.k1)
                           / (tf.tf + m.k1 * (1 - m.b + m.b * d.dl / m.avgdl))
                           * ln((m.n - t.nw + 0.5) / (t.nw + 0.5)) bm25s
                           from qtf
                               join t using (tid)
                               join tf using(tid)
                               join d using(did)
                               join m
                           where qtf.did in (?)
                    )
                    select did, sum(bm25s), text
                        from b join d using (did)
                        where bm25s > 0
                        group by did
                        order by 2 desc
                        limit (?);
                    """,
                    (qid, k),
                )
                ds = c.fetchall()
                c.close()

        d: dict = dict(ds=detuple(ds), q=self.retrieve_query(qid))

        pos_did = d["q"][2]
        neg_did = d["q"][3]
        pos_rr = 0.0
        neg_rr = 0.0
        pos_rank = 0
        neg_rank = 0

        for i, x in enumerate(d["ds"], start=1):
            did = x[0]
            if did == pos_did:
                pos_rank = i
                pos_rr = 1.0 / pos_rank
            if did == neg_did:
                neg_rank = i
                neg_rr = 1.0 / neg_rank

        d["k"] = k
        d["pos_rank"] = pos_rank
        d["neg_rank"] = neg_rank
        d["pos_rr"] = pos_rr
        d["neg_rr"] = neg_rr

        return d


def detuple(xs: list[tuple]) -> list[list]:
    return [list(x) for x in xs]


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
        level=getattr(
            logging,
            log_level.upper(),
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def main() -> None:
    setup_logging("debug")

    db_path = sys.argv[1]
    tsv_path = sys.argv[2]
    start_line_id = int(sys.argv[3])
    last_qid = int(sys.argv[4])

    ingest_flag = not Path(db_path).exists()

    np.random.seed(RANDOM_SEED)

    logging.info(f"Initializing bm25s model with LLM {LANGUAGE_MODEL_NAME!r}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    if ingest_flag:
        language_model = transformers.AutoModel.from_pretrained(LANGUAGE_MODEL_NAME).to(
            DEVICE
        )
    else:
        language_model = None

    bm = BM25S(
        language_model,
        tokenizer,
        db_path,
        k1=K1,
        b=B,
        embedding_size=EMBEDDING_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        device=DEVICE,
    )

    if ingest_flag:
        logging.info(
            f"Ingesting corpus {tsv_path=}, {start_line_id=} into {db_path=}..."
        )
        bm.ingest_corpus(tsv_path, start_line_id)

    logging.info(f"Running bm25s for 1..{last_qid}, {K=}...")
    pos_mrr_sum = 0.0
    neg_mrr_sum = 0.0
    cnt = 0
    for qid in (pbar := tqdm.trange(1, last_qid + 1)):
        d = bm.bm25s_by_qid(qid, K)
        cnt += 1
        pos_mrr_sum += d["pos_rr"]
        neg_mrr_sum += d["neg_rr"]
        pos_mrr = pos_mrr_sum / cnt
        neg_mrr = neg_mrr_sum / cnt
        pbar.set_description(f"+{pos_mrr:9.9f} -{neg_mrr:9.7f}")


if __name__ == "__main__":
    main()
