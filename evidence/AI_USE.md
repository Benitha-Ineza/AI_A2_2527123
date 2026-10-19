# AI Use Disclosure

**Assignment:** SWE 3513 – Artificial Intelligence, Individual Assignment 2  
**Project:** Musanze Harvest Intelligence Service  
**Student:** INEZA Benitha  
**Registration number:** 25/27123  
**Institutional email:** ug2527123@ines.ac.rw  
**Verification code:** AI-A2-7123

## 1. Use of ChatGPT

ChatGPT was used as a supporting research and learning resource for this assignment. It helped explain machine-learning concepts, explore suitable Python approaches, review implementation ideas, and adapt Python code examples to the requirements of the Musanze Harvest Intelligence Service.

AI assistance also supported code development, debugging, test preparation, and technical documentation. The student should verify that this description covers all AI-assisted development activities actually undertaken.

## 2. How AI was used

ChatGPT provided research explanations and guidance on synthetic data generation, data preprocessing, regression, classification, clustering, evaluation metrics, command-line prediction, FastAPI validation, and Python testing. Python code suggestions and examples were adapted to the project's required inputs, outputs, student seed, and verification code rather than copied as agricultural findings.

The project uses **synthetic data**, generated using the student-specific seed **7123**, rather than measurements from real farms. Model results therefore illustrate an academic decision-support workflow and are not independently validated farming recommendations.

## 3. Testing and verification

The following outcomes were observed in the project's PowerShell terminal or API responses:

| Check | Observed result |
|---|---|
| Automated tests | 8 passed, 1 warning (most recent reported run: 27.52 seconds) |
| Full pipeline | Completed for `ug2527123@ines.ac.rw` with seed `7123` |
| Generated dataset | 273 rows |
| Dataset SHA-256 | `759d98eee29266a8d54d7de3c50c4a24c4ea50d4d5b38bf7082ed4ba3cb3e014` |
| Model selection | Linear Regression; Logistic Regression; 2 clusters |
| Valid CLI prediction | Returned harvest, risk, probability, cluster, and `AI-A2-7123` |
| Invalid CLI prediction | Rejected missing `cooperative_support` |
| API | `/health`, `/model-info`, and `/docs` accessible |
| Valid API prediction | Returned a prediction through `POST /predict` |
| Invalid API prediction | Returned HTTP 422 for missing required field |

The single pytest warning concerned a Starlette/httpx deprecation notice, not a failing test. These checks do **not** establish that every possible input or deployment scenario was tested.

## 4. Student responsibility and limitations

The student is responsible for reviewing the source code, understanding the methodology, checking the report against generated artifacts, verifying the final submission files, and accurately presenting the demonstration. Any declaration must be reviewed and personally signed by the student.

AI-assisted outputs can contain errors. This disclosure does not claim that the system has been independently audited or validated on real agricultural data. No real-world decision should be made solely from its predictions.

## 5. Transparency

This disclosure is intended to make the use of AI assistance visible and accurate. It should be updated by the student if additional tools or forms of assistance were used or if any description above does not match what actually occurred.
