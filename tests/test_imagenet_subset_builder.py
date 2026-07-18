import types
import unittest
from collections import OrderedDict

from scripts.build_imagenet_flow_cache_subset import select_synsets


class ImageNetSubsetBuilderTest(unittest.TestCase):
    def test_stratified_selection_spans_manifest_class_order(self):
        by_synset = OrderedDict(
            (f"n{idx:08d}", list(range(10)))
            for idx in range(20)
        )
        args = types.SimpleNamespace(
            synsets_file="",
            synsets="",
            min_samples_per_class=10,
            num_classes=4,
            class_selection="stratified",
            seed=42,
        )

        selected = select_synsets(args, by_synset)
        selected_positions = [list(by_synset).index(synset) for synset in selected]

        self.assertEqual(len(selected), 4)
        self.assertTrue(0 <= selected_positions[0] < 5)
        self.assertTrue(5 <= selected_positions[1] < 10)
        self.assertTrue(10 <= selected_positions[2] < 15)
        self.assertTrue(15 <= selected_positions[3] < 20)
        self.assertEqual(selected, select_synsets(args, by_synset))

    def test_automatic_selection_filters_undersized_classes(self):
        by_synset = OrderedDict(
            [
                ("n00000001", [1, 2]),
                ("n00000002", [3, 4, 5]),
                ("n00000003", [6, 7, 8]),
            ]
        )
        args = types.SimpleNamespace(
            synsets_file="",
            synsets="",
            min_samples_per_class=3,
            num_classes=2,
            class_selection="first",
            seed=42,
        )

        self.assertEqual(
            select_synsets(args, by_synset),
            ["n00000002", "n00000003"],
        )


if __name__ == "__main__":
    unittest.main()
