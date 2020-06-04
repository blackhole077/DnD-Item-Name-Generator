# DnD-Item-Name-Generator
A deep learning approach to generating Dungeon and Dragons-themed item names.


## Introduction

Within a name, lies some inherent, intrinsic value. Whether the name is attached to a person, place, or even an object, the name itself can adjust the perception of what it's attached to. The fantastical, regardless of medium and origin, seems to make great use of this notion with named items. Whether it be the cursed Frostmourne [cite], or the mythological Excalibur [cite], named items bestowed some relevance, and thus meaning, to the world it was placed in.

In the context of games like Dungeons \& Dragons [cite], named items carry largely the same purpose. However, due to the open-ended style of story-telling that forms the crux of the game, these named items also provide chances for more world-building, exploration, and immersion.

## Motivation

Unfortunately, coming up with names isn't always a simple task. Nor is it always the intent to have intricately crafted tales associated with the weapons. Therefore, the task that this program is designed to tackle is simple: **Come up with "realistic" sounding fantasy item names.**

The use for this is equally simple. It's a tool for those seeking some spark of inspiration, or just a name to throw out at inquisitive minds. Regardless, it is designed with the idea that being able to quickly get something in mind for refinement (or not) is an important facet to world-building, and having a tool that can provide a variety of examples to work with is invaluable.

However, from a programming perspective, there lies one more motivation. To bring in the current research on similar topics (i.e. Text generation), combined with some sort of "Automated Quality Checking" to see if creating an self-scaling dataset for this particular task is possible.


## Dataset Details
The dataset currently is a text file of roughly 140K item names, with each entry separated by the newline character. Data was scraped from various game-related websites, where tables containing item names, among other things, were present. Furthermore, JSON dumps of items were parsed for item names for especially large games, such as World of Warcraft and Diablo 3.

The design decision behind compiling the dataset in this fashion is simple. Since Dungeons \& Dragons relies rather heavily on a given Dungeon Master's ability to generate unique item names, the number of "named" items that are unique is rather limited. Even including material from book series would not yield a sufficiently large dataset. Therefore, it made sense to augment through examples from other games that would still fall under the "Fantasy" genre of games. Furthermore, these games are likely drawn upon already by Dungeon Masters when creating item names, albeit not in the same fashion as this approach.

There does remain an issue of imbalance in the dataset, however, as a majority of the names are sourced from games made by Blizzard. This means that, to some extent, this generator will create names using a character embedding that's more similar to Blizzard-style games than any other property included in the dataset. Unfortunately, this issue isn't easily solvable without including a large amount of data from other games, so as of now it remains unsolved.

## Architecture Details

The current working architecture is a sequence of the following, based on the Crepe architecture proposed by Zhang et al \cite{zhang_text_2016}:
1. Input Layer
2. Embedding Layer
3. Convolutional Stack (1D), with Thresholded ReLU
4. MaxPooling1D Layer (but only for certain layers)
5. Fully Connected / Dense Stack

The rationale for this choice came from repeated attempts to utilize an LSTM-only architecture, with little success. Since the problem description is best suited for character-by-character generation, it seems that recent (meaning within last 5 years) research has shifted towards using some combination of Convolutional Neural Networks (cite papers) and LSTMs. Additionally, network paradigms like Generative Adversarial Networks (e.g., cite papers here) have shown promise in text generation. However, one major advantage of using Convolutional Neural Networks for this task is, unlike the other model architectures, it can be trained in a fairly short amount of time, at the cost of losing some of the temporal relationships that are otherwise preserved in LSTM-type networks.

As for why this architecture was chosen, there isn't a specific reason. Simply put, it was one of the first architectures that I was able to understand well enough to take a Keras implementation (found here: \cite{jabreel_mhjabreelcharcnn_keras_2020}) and re-work it for my dataset. Given the architecture was proposed in 2015, there are likely better performing architectures now, which I would suggest looking into, should you be interested in creating a better performing version of this model.

## Notes for Self
- There seems to be a couple entries **\\xa0**, **\\ufeff** that should not be present, which means re-scrubbing will be necessary.
- The name generator being finished, while great, is not the end of this project. There will need to be an additional layer of complexity added to handle what will hopefully pan out as "Automated QC", probably in the form of some kind of GAN-like structure. Currently unclear if I can use the existing architecture, or if an overhaul will be necessary.
