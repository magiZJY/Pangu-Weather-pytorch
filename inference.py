from model import *

def Inference(input, input_surface, forecast_range):
  '''Inference code, describing the algorithm of inference using models with different lead times.
  PanguModel24, PanguModel6, PanguModel3 and PanguModel1 share the same training algorithm but differ in lead times.
  Args:
    input: input tensor, need to be normalized to N(0, 1) in practice
    input_surface: target tensor, need to be normalized to N(0, 1) in practice
    forecast_range: iteration numbers when roll out the forecast model
  '''

  PanguModel24 = LoadModel(ModelPath24)
  PanguModel6 = LoadModel(ModelPath6)
  PanguModel3 = LoadModel(ModelPath3)
  PanguModel1 = LoadModel(ModelPath1)

  weather_mean, weather_std, weather_surface_mean, weather_surface_std = LoadStatic()

  input_24, input_surface_24 = input, input_surface
  input_6, input_surface_6 = input, input_surface
  input_3, input_surface_3 = input, input_surface

  output_list = []

  # Note: the following code is implemented for fast inference of [1,forecast_range]-hour forecasts -- if only one lead time is requested, the inference can be much faster.
  for i in range(forecast_range):
    # switch to the 24-hour model if the forecast time is 24 hours, 48 hours, ..., 24*N hours
    if (i+1) % 24 == 0:
      # Switch the input back to the stored input
      input, input_surface = input_24, input_surface_24

      # Call the model pretrained for 24 hours forecast
      output, output_surface = PanguModel24(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

      # Stored the output for next round forecast
      input_24, input_surface_24 = output, output_surface
      input_6, input_surface_6 = output, output_surface
      input_3, input_surface_3 = output, output_surface

    # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
    elif (i+1) % 6 == 0:
      # Switch the input back to the stored input
      input, input_surface = input_6, input_surface_6

      # Call the model pretrained for 6 hours forecast
      output, output_surface = PanguModel6(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

      # Stored the output for next round forecast
      input_6, input_surface_6 = output, output_surface
      input_3, input_surface_3 = output, output_surface

    # switch to the 3-hour model if the forecast time is 3 hours, 9 hours, ..., 6*N + 3 hours
    elif (i+1) % 3 ==0:
      # Switch the input back to the stored input
      input, input_surface = input_3, input_surface_3

      # Call the model pretrained for 3 hours forecast
      output, output_surface = PanguModel3(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

      # Stored the output for next round forecast
      input_3, input_surface_3 = output, output_surface

    # switch to the 1-hour model
    else:
      # Call the model pretrained for 1 hours forecast
      output, output_surface = PanguModel1(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

    # Stored the output for next round forecast
    input, input_surface = output, output_surface

    # Save the output
    output_list.append((output, output_surface))
  return output_list

