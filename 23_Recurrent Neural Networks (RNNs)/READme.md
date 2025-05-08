# Topic 23: Recurrent Neural Networks (RNNs)

## 1. Overview

Recurrent Neural Networks (RNNs) are a class of artificial neural networks specifically designed to process **sequential data** or time-series data. Unlike feedforward neural networks (ANNs, CNNs) which assume inputs are independent, RNNs have internal "memory" that allows them to persist information from previous inputs in the sequence to influence the processing of current and future inputs. This makes them suitable for tasks where context and order are crucial.

**Examples of Sequential Data:**
* Text (sequences of words or characters)
* Speech (sequences of audio features)
* Time series (e.g., stock prices, weather data over time)
* Video (sequences of image frames)

## 2. Core Idea: Loops and Hidden States

* **Recurrent Connection (Loop):** The defining feature of an RNN is a loop within its architecture. The output of a layer (or a specific hidden state) at one time step is fed back as an input to the same layer (or cell) at the next time step.
* **Hidden State ($h_t$):** This is the "memory" of the RNN. At each time step $t$, the hidden state $h_t$ is computed based on the current input $x_t$ and the previous hidden state $h_{t-1}$.
    `h_t = f(W_xh * x_t + W_hh * h_{t-1} + b_h)`
    (where $f$ is an activation function like tanh or ReLU, and $W$ terms are weights, $b$ is bias).
* **Parameter Sharing:** The same set of weights ($W_{xh}, W_{hh}$) and biases are used across all time steps for processing the sequence. This allows RNNs to handle sequences of varying lengths and generalize learned patterns across different positions in the sequence.
* **Output ($ŷ_t$):** An output can be produced at each time step, typically derived from the hidden state:
    `ŷ_t = g(W_hy * h_t + b_y)`
    (where $g$ is an output activation function).

## 3. Different RNN Architectures

RNNs can be configured for various sequence processing tasks:
* **One-to-Many:** Single input, sequence output (e.g., image captioning).
* **Many-to-One:** Sequence input, single output (e.g., sentiment analysis, text classification).
* **Many-to-Many (Synchronized):** Sequence input, sequence output where each output corresponds to an input (e.g., part-of-speech tagging, video frame classification).
* **Many-to-Many (Delayed - Encoder-Decoder):** Sequence input, sequence output where output generation starts after processing the entire input (e.g., machine translation).

## 4. Training RNNs: Backpropagation Through Time (BPTT)

* RNNs are trained using a modified version of backpropagation called **Backpropagation Through Time (BPTT)**.
* The RNN is "unrolled" for the length of the input sequence, creating a deep feedforward network where each layer corresponds to a time step.
* The error is calculated (e.g., at the end of the sequence or summed over time steps).
* Gradients are then propagated backward through this unrolled network.
* Since weights are shared across time steps, the gradients for these shared weights are accumulated (summed) across all time steps before an update is made.

## 5. The Vanishing/Exploding Gradients Problem

Training simple (vanilla) RNNs on long sequences is challenging due to:
* **Vanishing Gradients:** As gradients are propagated back through many time steps, they can become extremely small (diminish exponentially). This makes it difficult for the network to learn long-range dependencies, as earlier time steps receive very small updates.
* **Exploding Gradients:** Conversely, gradients can become extremely large (grow exponentially), leading to unstable training and numerical overflows (weights becoming NaN or infinity).
    * **Mitigation for Exploding Gradients:** Gradient Clipping (scaling down gradients if their norm exceeds a threshold).

The vanishing gradient problem is more fundamental and was a major motivation for developing more advanced recurrent architectures.

## 6. Long Short-Term Memory (LSTM)

LSTMs are a type of RNN specifically designed to overcome the vanishing gradient problem and effectively learn long-range dependencies.

* **Core Idea:** LSTMs use a **gated cell structure** to regulate the flow of information.
* **Key Components:**
    1.  **Cell State ($C_t$):** Acts as the "memory conveyor belt," allowing information to flow through time with minimal modification, regulated by gates.
    2.  **Forget Gate ($f_t$):** A sigmoid layer that decides what information to discard from the previous cell state ($C_{t-1}$).
    3.  **Input Gate ($i_t$):** A sigmoid layer that decides which new values to update in the cell state. It works with a tanh layer that creates candidate values ($\tilde{C}_t$).
        `C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`
    4.  **Output Gate ($o_t$):** A sigmoid layer that decides what parts of the current cell state ($C_t$) to output as the hidden state ($h_t$) for the current time step.
        `h_t = o_t * tanh(C_t)`
* **Benefit:** The gates allow LSTMs to selectively add, remove, or pass through information in the cell state, preserving relevant information over long sequences.

## 7. Gated Recurrent Units (GRUs)

GRUs are a newer, simpler alternative to LSTMs that also address the vanishing gradient problem.

* **Core Idea:** GRUs also use gates but have a more streamlined architecture than LSTMs. They combine the forget and input gates into a single **update gate** and merge the cell state and hidden state.
* **Key Components:**
    1.  **Update Gate ($z_t$):** A sigmoid layer that determines how much of the previous hidden state ($h_{t-1}$) to keep and how much of the new candidate hidden state ($\tilde{h}_t$) to incorporate.
    2.  **Reset Gate ($r_t$):** A sigmoid layer that decides how much of the previous hidden state to forget when computing the candidate hidden state.
    3.  **Candidate Hidden State ($\tilde{h}_t$):** Proposed new hidden state based on the current input and a *reset* version of the previous hidden state.
    4.  **Final Hidden State ($h_t$):** A linear interpolation between the previous hidden state $h_{t-1}$ and the candidate hidden state $\tilde{h}_t$, controlled by the update gate $z_t$.
        `h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t`

## 8. LSTM vs. GRU

* **Complexity:** GRUs have fewer parameters and gates than LSTMs, making them computationally slightly simpler and potentially faster to train.
* **Performance:** Both are highly effective at capturing long-range dependencies. Performance is often comparable and task-dependent. LSTMs might have a slight edge on tasks requiring very long memory or more complex gating, while GRUs can be more efficient. It's common to try both.

## 9. Applications of RNNs, LSTMs, and GRUs

* **Natural Language Processing (NLP):** Machine translation, sentiment analysis, text generation, question answering, speech recognition, part-of-speech tagging.
* **Time Series Analysis:** Stock price prediction, weather forecasting, anomaly detection in sensor data.
* **Video Analysis:** Activity recognition, video captioning.
* **Music Generation.**

## 10. Implementation with PyTorch

* **Core Modules:**
    * `torch.nn.RNN`: For simple RNN cells.
    * `torch.nn.LSTM`: For Long Short-Term Memory cells.
    * `torch.nn.GRU`: For Gated Recurrent Unit cells.
* **Key Parameters for `nn.LSTM` / `nn.GRU`:**
    * `input_size`: Number of expected features in the input $x_t$.
    * `hidden_size`: Number of features in the hidden state $h_t$.
    * `num_layers`: Number of recurrent layers (for stacked RNNs/LSTMs/GRUs).
    * `batch_first=True`: If `True`, input and output tensors are provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
    * `dropout`: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer.
    * `bidirectional=True`: If `True`, becomes a bidirectional RNN/LSTM/GRU, processing the sequence in both forward and backward directions. The output hidden size will be `2 * hidden_size`.
* **Data Preparation:** Input data for RNNs typically needs to be shaped as `(batch_size, sequence_length, feature_size)` if `batch_first=True`.
* **Hidden State Initialization:** The initial hidden state (and cell state for LSTM) usually needs to be provided or is initialized to zeros.

---