# bm25s
BM25 model in token and semantic domains

## Database schema

### Metadata

```sql
CREATE TABLE m(n, avgdl, k1, b, s);
```

* `n` - total number of documents in collection
* `avgdl` - average document length in tokens
* `k1` - coefficient k1, normally in [1.2, 2.0]
* `b` - coefficient b, normally =0.75
* `s` - coefficient s: semantic (s) vs. token (1 - s) domain 

Example: 
```
sqlite> select * from m;
20000|81.01685|1.2|0.75|0.75
```

### Documents

```sql
CREATE TABLE d(did primary key, text, dl);
```

* `did` - document id (did > 0: pos docs, did < 0: neg docs)
* `text` - document text (not parsed)
* `dl` - document length in tokens

Example: 
```
sqlite> select * from d limit 3;
-12|However, you still need ... higher niacin requirements).|47
13|The blood levels will ... can also help.|102
-13|Low hemoglobin, high ... fish, nuts and avocados.|134
```

### Queries

```sql
CREATE TABLE q(did primary key, text, pos_did, neg_did);
```

* `did` - query id
* `text` - query text (not parsed)
* `pos_did` - the id of the “positive” doc, i.e., the document that should be found in response to this query
* `neg_did` - the id of the “negative” doc, i.e., the document that should not be found in response to this query

Example:
```
sqlite> select * from q limit 3;
12|what are some foods pregnant women avoid|12|-12
13|how long does blood take to replenish afet lossof blood|13|-13
14|what is the genus of the weeping willow tree|14|-14
```

### Tokens

```sql
CREATE TABLE t(tid primary key, token, nw);
```

* `tid` - token id (tid > 0: token domain, tid < 0: semantic domain)
* `token` - token text
* `nw` - number of documents in collection containing this token

Example:
```
sqlite> select * from t limit 5;
1104|of|15128
11019|ca|411
15475|##ffe|44
2042|##ine|330
1219|during|637
```

### Token Frequency

```sql
CREATE TABLE tf(tid, did references d(did), tf, primary key (tid, did));
```

* `tid` - token id (tid > 0: token domain, tid < 0: semantic domain)
* `did` - document id
* `tf` - token frequency: number of times token `tid` appears in document `did` 

Example:
```
sqlite> select * from tf limit 5;
1284|1|1
1274|1|1
28198|1|3
1204|1|1
1221|1|1
```

### Query Tokens

```sql
CREATE TABLE qt(tid primary key, token, nw);
```

### Query Token Frequency

```sql
CREATE TABLE qtf(did references q(did), tid, tf, primary key (did, tid));
```

## BM25S

The following query implements BM25S search:

TODO: consider taking into account qtf.tf, i.e., when the same token occur in the query multiple times

```sql
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
       where qtf.did in ({qid})
)
select did, sum(bm25s), text
    from b join d using (did)
    where bm25s > 0
    group by did
    order by 2 desc
    limit ({k});
```


## Proposed schema for PCP

```sql
CREATE TABLE metadata(n, avgdl, k1, b, s);
CREATE TABLE documents(did primary key, content_id, dl);
CREATE TABLE tokens(tid primary key, token, nw);
CREATE TABLE token_freq(tid, did references documents(did), tf, primary key (tid, did));
CREATE TEMPORARY TABLE query(tid primary key, tf);
```
