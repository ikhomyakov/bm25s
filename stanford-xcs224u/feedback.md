## Final paper

Great work! This was a well written report with a couple areas for improvement. In particular, 1) improving clarity and detail in describing your method (this is what you're propsing in the paper so it should take a main stage); 2) Avoid having related work as a Lit Review; and 3) Including more thorough experimental analysis. Overall all though great work throughout the course!

Here are some thoughts I collected while reading your report. One that does stand out is that I am not very convinced by this approach. I saw in your email some questions around whether this approach my for example beat ColBERT. I would be very surprised by this. I am happy to discuss more offline!

- You dive a bit too quickly into the nitty gritty of your method in the Intro. It would be great to do a bit more "scene" setting in terms of what you mean by proposing a novel "vocabulary in the semantic space.”

- You should not go into this much detail in the related work. As is recommended here: https://github.com/cgpotts/cs224u/blob/main/projects.md#related-work, you generally want to keep this to 1.5/2 columns. in the end it is your work you want people to focus on.

- I would recommend discussing your approach for creating the new "semantic" vocabulary in the "Models` section. This is in effect your Model!

- It would be nice to include a figure describing your approach!

- I must say I don't know if I fully buy into this approach. It seems that the "semantic" tokens you're creating are just a new representation of the original BERT tokens. I don't quite see how this leads to new information, other than potentially having more tokens to capture slightly more fine-grained "word forms." I guess a question I would have is imagine the "semantic token" vocab was exactly the same size as the BERT vocab - 

	1. do we expect LSH to effectively represent a 1-to-1 function? and 

	2. if this is roughly the case then won't the TF-IDF matrices used for BM25 and BM25-S be very similar? 

I guess a question I have is are we essentially doing BM25 with BERT tokens?

- Your experimental results are a bit weak. You should always compare against baselines. Moreover, here it would be beneficial to experiment over different formulations of your approach.


## Experimental protocol

Hypothesis: Very interesting idea! I think the idea of mapping to a novel semantic vocabulary is very interesting, and potentially challenging; however, it sounds like you are down a very interesting direction!

Datasets: Great choices of datasets.

Metrics: One thing I would also look to compare is latency and memory. This remember is one of your driving motivations. How BM25 can effectively be both ultra efficient and still performant.

Models: A big part of your method is the use of BM25 over dense embeddings. To me this is really the modeling approach you're proposing and should take a main stage in the description of your approach.

General reasoning: Very interesting methodology and intuition.

Summary of progress so far: You have clearly put a lot of great thought into this work! There definitely seem to be some challenges here and potentially unanswered questions. However, remember that it is okay in this project not to end up with the best possible solution. The journey and work itself can lead to really interesting insights. Feel free to reach out to discuss more!


## Lit review

Great work. Improving IR in a holistic way is certainly a very prudent topic.

Here you could have expanded a bit more on the Compare and Contrast section. This section is very helpful for both identifying holes in the research field + drawing inspiration across multiple ideas.

For you future directions I really like the two pronged approach of exploring:

    1. Better methods by ideally fusing methods. This may be hard to get working but will be very interesting!

    2. Focusing on evaluation. This will be a great way to supplement your report.


Description details of papers are insufficient. Or a coherent description of papers regarding the task is missing. Or, incoherent set of papers.
More detailed compare and contrast is expected.

## Assignment 3

In practice, I would recommend against using the same model you want to improve to generate the few-shot samples. You really want to make sure these samples are "gold" samples. So in this case I would worry about providing potentially incorrect few shots.

## Assignment 2

Good work. One suggestion that I might have is to experiment more with the prompt-engineering involved in combining the 2 first stages. I think this is a reasonable first pass - concatenating the answers - but the model may have a bit of trouble exactly understanding what you're trying to tell it. For example, you consider trying to do something like A) include context then B) have the question and then C) say to the model "here are a couple possible answers we have brainstormed ..." This type of approach connects to complex prompt strategies such as Chain of Thought / Tree of Thoughts, where the model spends time thinking outloud / proposing multiple solutions before settling on one.

## Assignment 1

Exceptional System, Description, and/or Highly Innovative Design
I really appreciate the unique approach here - even if it did not lead to the best results. It is interesting how the baseline / first step for everyone now days is to fine-tune some pre-trained model. But a somewhat forgotten baseline is the good old Nearest-Neighbor classifier (which you effectively use). This seems honestly a reasonable approach, especially with a pre-trained model.

One concern, is certainly how this approach handles out of domain data. Additionally, it would be interesting to experiment with non-BERT models that are better sentence encoders - though I like how you do a mean pooling of the tokens rather than take the [CLS] token. Since BERT is a Cross-Encoder model, the paper talks about how from pre-training this [CLS] token cannot be meaningfully interpreted as a sentence/document encoding. one thing to watch out for is doing mean pooling over the <pad> token. We likely want to ignore these.

Another interesting thing to try would be to use true k Nearest Neighbors with e.g. majority voting rather than centroids. Anyways, nice stuff.
