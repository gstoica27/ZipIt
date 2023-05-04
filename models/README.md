# Models
Here we provide instructions for adding models to the repository. It is generally very straightforward and only takes two steps.

#### Create architecture file
The first step involves simply creating a file with your model's architecture. You are free to name it anything, but may follow our current simple naming convention if desired. Additionally, you can include other util files for the model as we have for our SinGAN implementation.

#### Make config parser recognize your model.
The second and last step involves making our config parser recognize your model file. To do this, first add an elif to the "prepare_models" function in the utils, that checks whether the model name provided in the experiment config matches the one for your new model. Second, add a function called "prepare_YOURMODEL" that will handle all model preparation. You can see our currently implemented "prepare_modelX" functions for examples, but all this function needs to contain is the import from your model file, as well the capability to load your pretrained base models, and initialize a "new" model which will act as the merged model. You may follow the template provided by the other prepare_modelX functions. 