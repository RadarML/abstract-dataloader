# Abstract Dataloader: Dataloader Not Included

[![pypi version](https://img.shields.io/pypi/v/abstract-dataloader.svg)](https://pypi.org/project/abstract-dataloader/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/abstract-dataloader)
![PyPI - Types](https://img.shields.io/pypi/types/abstract-dataloader)
![GitHub](https://img.shields.io/github/license/RadarML/abstract-dataloader)
[![CI](https://github.com/RadarML/abstract-dataloader/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/abstract-dataloader/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/abstract-dataloader)

## What is the Abstract Dataloader?

The **abstract dataloader** (ADL) is a minimalist [specification](https://radarml.github.io/abstract-dataloader/spec/) for creating composable and interoperable dataloaders and data transformations, along with [abstract template implementations](https://radarml.github.io/abstract-dataloader/abstract/) and reusable [generic components](https://radarml.github.io/abstract-dataloader/generic/).

![Abstract Dataloader Overview](https://radarml.github.io/abstract-dataloader/diagrams/overview.svg)

The ADL's specifications and bundled implementations lean heavily on generic type annotations in order to enable type checking using static type checkers such as [mypy](https://mypy-lang.org/) or [pyright](https://microsoft.github.io/pyright/) and runtime (dynamic) type checkers such as [beartype](https://github.com/beartype/beartype) and [typeguard](https://github.com/agronholm/typeguard).

> [!TIP]
> Since the abstract dataloader uses python's [structural subtyping](https://typing.python.org/en/latest/spec/protocol.html) - `Protocol` - feature, the `abstract_dataloader` is not a required dependency for using the abstract dataloader! Implementations which follow the specifications are fully interoperable, including with type checkers, even if they do not have any mutual dependencies - including this library.

For detailed documentation, please see the [project site](https://radarml.github.io/abstract-dataloader/).

## Why Abstract?

Loading, preprocessing, and training models on time-series data is ubiquitous in machine learning for cyber-physical systems. However, unlike mainstream machine learning research, which has largely standardized around "canonical modalities" in computer vision (RGB images) and natural language processing (ordinary unstructured text), each new setting, dataset, and modality comes with a new set of tasks, questions, challenges - and data types which must be loaded and processed.

This poses a substantial software engineering challenge. With many different modalities, processing algorithms which operate on the power set of those different modalities, and downstream tasks which also each depend on some subset of modalities, two undesirable potential outcomes emerge:

1.  Data loading and processing components fragment into an exponential number of incompatible chunks, each of which encapsulates its required loading and processing functionality in a slightly different way. The barrier this presents to rapid prototyping needs no further explanation.
2.  The various software components coalesce into a monolith which nominally supports the power set of all functionality. However, in addition to the compatibility issues that come with bundling heterogeneous requirements such as managing "non-dependencies" (i.e. dependencies which are required by the monolith, but not a particular task), this also presents a hidden challenge in that by support exponentially many possible configurations, such an architecture is also exponentially hard to debug and verify.

However, we do not believe that these outcomes are a foregone conclusion. In particular, we believe that it's possible to write "one true dataloader" which can scale while maintaining intercompability by **not writing a common dataloader at all** -- but rather a common specification for writing dataloaders. We call this the **"abstract dataloader"**.

## Setup

While it is not necessary to install the `abstract_dataloader` in order to take advantage of ADL-compliant components, installing this library provides access to `Protocol` types which describe each interface, as well as generic components which may be useful for working with ADL-compliant components.

```sh
pip install abstract-dataloader
```
