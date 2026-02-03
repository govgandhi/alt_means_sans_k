// skipgram_ops.cpp
#include <torch/extension.h>
#include <stdexcept>
#include <tuple>

// Computes the positive and negative scores for the skip-gram model.
// sim_type: 0 -> dot; 1 -> euclidean; 2 -> cosine.
std::tuple<at::Tensor, at::Tensor> skipgram_forward(
    const at::Tensor &center,       // (B, D)
    const at::Tensor &pos_context,  // (B, D)
    const at::Tensor &neg_context,  // (B, K, D)
    int sim_type,
    double beta                     // scaling factor for Euclidean similarity
) {
    at::Tensor pos_scores, neg_scores;
    if (sim_type == 0) {
        // DOT PRODUCT SIMILARITY
        // Positive: dot product along the embedding dimension.
        pos_scores = (center * pos_context).sum(1);
        // Negative: for each sample in the batch, compute dot products with negatives.
        auto center_unsqueezed = center.unsqueeze(2);  // (B, D, 1)
        neg_scores = at::bmm(neg_context, center_unsqueezed).squeeze(2);  // (B, K)
    } else if (sim_type == 1) {
        // EUCLIDEAN SIMILARITY
        // For positive pair: negative Euclidean distance scaled by beta.
        pos_scores = -at::norm(center - pos_context, 2, 1) * beta;
        // For negatives: expand center to (B, K, D) and compute norm.
        auto center_exp = center.unsqueeze(1).expand_as(neg_context);
        neg_scores = -at::norm(center_exp - neg_context, 2, 2) * beta;
    } else if (sim_type == 2) {
        // COSINE SIMILARITY
        auto pos_dot = (center * pos_context).sum(1);
        auto center_norm = at::norm(center, 2, 1);
        auto pos_context_norm = at::norm(pos_context, 2, 1);
        pos_scores = pos_dot / (center_norm * pos_context_norm + 1e-8);
        
        auto center_unsqueezed = center.unsqueeze(2);  // (B, D, 1)
        auto neg_dot = at::bmm(neg_context, center_unsqueezed).squeeze(2);  // (B, K)
        auto neg_context_norm = at::norm(neg_context, 2, 2);  // (B, K)
        neg_scores = neg_dot / (center_norm.unsqueeze(1) * neg_context_norm + 1e-8);
    } else {
        throw std::runtime_error("Unknown similarity type");
    }
    return std::make_tuple(pos_scores, neg_scores);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("skipgram_forward", &skipgram_forward, "Skip-gram forward (dot, euclidean, cosine)");
}
