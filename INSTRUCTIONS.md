# CS 898BA: Course Project Requirement Synopsis
Instructions from `https://github.com/codyfarlow1/CS898BA-PROJECT.git`

Contact me with questions, comments, or concerns: cody.farlow@wichita.edu
## Project Overview
The semester-long project for CS 898BA requires you to select, design, and implement an application of computer vision to solve a real-world problem. This project serves as a comprehensive application of the theoretical concepts covered throughout the course.

Students are expected to conduct a reasonably thorough review of existing literature and condense those foundational ideas to inform their own system design. Following design approval, you will implement the solution and perform hyperparameter optimization tailored for deep learning architectures.

Crucially, standard deep learning approaches must be paired with robust domain engineering. Image analysis and processing techniques should not be overlooked; relying solely on simple data augmentation will not satisfy the processing requirements for this project.

As this is an online course, all project milestones, deliverables, and presentations will be conducted completely asynchronously.

---

## Project Milestones & Required Content

### 1. Project Pitch & Scope Approval 
* **Objective:** Proposal of your project concept, target dataset, and core methodology.
* **Required Presentation Content:**
    * **Literature Review (Preliminary):** A brief survey of existing approaches to your chosen problem.
    * **Alternative Design Options:** At least two distinct technical approaches or model architectures that could solve the problem.
    * **Choice Justification:** A clear rationale explaining why your chosen approach is superior to the alternatives (e.g., computational efficiency, accuracy constraints, data availability).
    * **Image Processing Strategy:** Outline of the initial image analysis and processing steps beyond basic data augmentation.
* **Approval Gate:** You must receive explicit instructor approval on your project pitch before beginning any development work.
* **Scope Adjustment:** During this review, the instructor may add or reduce complexity to ensure the project is appropriately scaled for a single-semester graduate-level course. If a project is deemed too straightforward, additional requirements may be introduced; if it is too broad, the scope may be scaled back.

### 2. Midterm Progress Report & Presentation 
* **Objective:** Demonstrate initial progress, baseline validation, and proof of concept.
* **Required Presentation Content:**
    * **Condensed Lit Review:** An updated, synthesized summary of the core papers influencing your implementation.
    * **Data Pipeline & Processing:** Demonstration of your image processing, analysis, and data preparation pipeline (proving it goes beyond simple augmentation).
    * **Baseline Implementation:** Preliminary results from your initial model execution.
    * **Roadblocks & Pivots:** Discussion of unexpected technical challenges and how you adjusted your design or alternative options to address them.

### 3. Final Presentation & Demonstration 
* **Objective:** Showcase the completed, optimized application and defend your engineering decisions.
* **Required Presentation Content:**
    * **Final Architecture & Design:** A complete walkthrough of the finalized computer vision pipeline.
    * **Hyperparameter Optimization:** Detailed documentation of your hyperparameter tuning process for the deep learning models and their specific impact on performance.
    * **Image Analysis Evaluation:** Analysis of how your pre-processing and image processing steps directly contributed to the model's success.
    * **Results & Metrics:** A rigorous evaluation of your solution using appropriate computer vision metrics, compared against your baseline.
    * **Virtual Demonstration:** A live or recorded video demonstration showcasing your computer vision pipeline successfully processing data in real-time or batch format.

---

## Submission Items
### 1. Presentation Recording Format:

All presentations should be recorded in PowerPoint or your preferred software with both video and audio.

### 2. Milestone Deliverables:
* Submission of each presentation should include:

   * **Repository Link:** The git link of your project, which follows the git instructions.
        * **i. Pitch Repository:** Your repository should include items 1-3 of the git instructions, along with an empty (or research prompt-only) AI log.
    * **Video Access:** The YouTube link to your presentation video (either unlisted or public, DO NOT MAKE IT PRIVATE).
        * **i. Separate Demonstration Option:** You can send a different link to a demo video for the final presentation if you want.
    * **Visual Materials:** The identical slide deck that is used in the presentation video.

---

## Git Instructions
### 1. Repository Creation:
* Create a git repository for a Python codebase named `FirstnameLastname-CS898BA-Project`.
### 2. Initial Commit:
* Create an initial commit with a “Hello World!” Script.
### 3. AI Accountability:
* Keep track of all AI usage in a file named `AI_Log.md`.
    * **a. Log Content Requirements:** Each entry should include the full prompt, the date and time the prompt was issued, the AI tool used to enter the prompt, a synopsis of the result, and any relevant design or code changes affected by the result.
    * **b. Sample Reference Table:** Here is a very silly example:
        | Date and Time | Prompt | Tool | Response Synopsis | Change |
        | :--- | :--- | :--- | :--- | :--- |
        | 05/05/2026<br>12:00 AM | When is Cinco De Mayo? | ChatGPT | Today | Changed dinner plans. |
        | 05/10/2026<br>3:15 PM | I totally forgot to buy my mom a gift and she is coming over!!! Please make an image of a handwritten letter telling her how much I love her. | Gemini | Happy Mother’s Day letter and scolding me for being terrible/ungrateful. | Set reminders of Mother’s Day and felt deep shame. |
### 4. Incremental Version Control:
* Perform incremental commits with updated code for each change with a quality commit message.
### 5. Documentation Standard:
* Ensure you include a comprehensive ReadMe file that explains the code, outlines setup and execution steps, includes your results, and provides any relevant discussion.