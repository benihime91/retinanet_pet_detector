# retinanet_pet_detector

Using a `Retinanet` to detect faces of `cats & dogs`.

Create a `PetDetector` which can detect the `faces` of cats & dogs in Images using my implementation of [Retinanet](https://github.com/benihime91/pytorch_retinanet).

## **Dataset used**:

`The Oxford-IIIT Pet Dataset` which can be found [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## **TODO**:

- [x] Parse the data and convert it to a managable format ex: CSV.
- [x] Finish [Retinanet Project](https://github.com/benihime91/pytorch_retinanet) first.
- [x] Train the Network.
- [x] Create WebApp using `StreamLit`.
- [x] Deploy WebApp . `(Removed due to cost constraints)`
- [ ] Notebooks & Scripts for Train. 

## **Usage**:

- Install [python3](https://www.python.org/downloads/)
- Install dependencies
  ```bash
  git clone https://github.com/benihime91/retinanet_pet_detector.git
  cd retinanet_pet_detector
  pip install -r requirements.txt
  ```
- Run app
  ```bash
  streamlit run app.py
  ```