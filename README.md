# MoToRev - Model-Driven Tourism Recommender System

![Java](https://img.shields.io/badge/Java-17-blue)
![Eclipse Modeling](https://img.shields.io/badge/Eclipse_Modeling-EMF%2C_Epsilon-green)
![Apache Mahout](https://img.shields.io/badge/Apache_Mahout-0.14.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![MDE](https://img.shields.io/badge/Approach-Model_Driven_Engineering-blueviolet)

## About MoToRev

MoToRev is a Model-Driven Engineering (MDE) approach for developing Tourism Recommender Systems (TRSs). This repository contains the implementation of our novel framework that addresses the key challenges in TRS development:

## Key Innovations

- **Model-Driven Approach**: Encodes TRS knowledge as reusable modelling artefacts
- **Hybrid Recommendation**: Combines collaborative, content-based, and hybrid filtering
- **Automated Pipeline**: From data models to recommendations and UI generation
- **RASTA Project Proven**: Validated in sustainable tourism scenarios
- **Domain Adaptable**: Flexible architecture for diverse tourism applications

## Repository Structure
```
MoToRev/
├── src/
│ ├── main/
│ │ ├── java/genericRecommenderSystem/ - Core recommendation engine
│ │ │ ├── Main.java - Entry point with the recommendation logic
│ │ │ └── ... - Supporting classes
│ │ ├── Models/ - EMF-based domain models
│ │ │ ├── recommendersystemGeneric.ecore - Metamodel
│ │ │ ├── domain.ecore - Domain concepts
│ │ │ ├── EOL_scripts/ - Model transformation scripts
│ │ │ └── ... - Instance models
│ │ └── resources/ - Configuration files
├── target/ - Build output
└── pom.xml - Maven configuration
```

## Getting Started

### Prerequisites

- Java 17+
- Maven 3.6+
- Eclipse Modeling Tools (recommended for development)
- Apache Mahout 0.14.0

### Installation

```bash
git clone https://github.com/ricksonsimioni/MoToRev-Leveraging-Interactive-Modeling-Artifacts-to-Assist-Tourism-Recommender-Systems-Development.git
cd MoToRev-Leveraging-Interactive-Modeling-Artifacts-to-Assist-Tourism-Recommender-Systems-Development/
mvn clean install
