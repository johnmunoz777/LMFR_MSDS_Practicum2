# Live Member Face Recognition Deployment

## Project Description

The purpose of this project is to develop a more efficient way to verify memberships and grant access to members without unnecessary delays.  
This issue exists in places like Costco, sporting events, and other venues where long queues form just to validate entry credentials.  
The goal is to eliminate long wait times by allowing members to gain access seamlessly through facial recognition, reducing the need for manual verification.

<!-- 3‑column CSS grid. Fill in as many <figure> blocks as you have images. -->
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; justify-items: center;">
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use.png" alt="Members Example 1" width="200" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;">200×200 px</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use_two.jpg" alt="Members Example 2" width="200" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;">200×200 px</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use_three.jpg" alt="Members Example 3" width="200" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;">200×200 px</figcaption>
  </figure>
  <!-- Duplicate or add more <figure> blocks here for a full 3×3 grid -->
</div>

## Project Proposal

To solve this problem, I developed a real‑time face recognition system using computer vision.  
In [Practicum 1: YOLOv8 Model](https://github.com/johnmunoz777/LMFD_MSDS_Practicum) I built the initial face recognition model.  
The purpose of [Practicum 2: Deployment Demo](https://facedetectiondemo.com/) was to deploy this model so that:
- Members can add themselves to the system.
- Employees can validate members in real time via webcam.
