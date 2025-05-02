## Welcome to the repo of Estimating the predictability of questionable open-access journals.

## Authors
Han Zhuang, Lizhen Liang, and Daniel Acuna <sup>\+</sup>

<sup>\+</sup>Corresponding author: daniel.acuna@colorado.edu

## Abstract
Deceptive publishing practices in scientific journals are a significant risk to the integrity of global knowledge dissemination. Traditional methods of identifying legitimate open-access journals rely heavily on manual verification, which is time-consuming, error-prone, and difficult to adapt. In this study, we explore the potential of artificial intelligence to systematically identify questionable journals by analyzing their websites' design, content, and publication metadata. Our approach, validated against more than 15,000 manually annotated journals and a novel dataset annotated by experts, demonstrates notable accuracy. First, we identify several predictors of journal legitimacy currently absent from standard evaluation protocols. Second, our AI model identifies more than one thousand potentially questionable journals, which have published hundreds of thousands of articles. Third, the analysis of these articles reveals a concerning trend: they are cited millions of times, acknowledged in NSF and NIH grants, and are gaining traction among researchers in developing nations. Finally, we discuss how AI can be a supportive tool in monitoring and promoting the integrity of the scientific publication ecosystem. 

The data folder contains datasets to reproduce the results in our paper. The journal_website_processing folder contains the code to scrape journal website and produce features to predict questionable journals. The reproduce_results folder contains jupyter notebook to reproduce the results in our paper.
