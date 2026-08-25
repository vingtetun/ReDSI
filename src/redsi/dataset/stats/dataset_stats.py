from dataclasses import dataclass

from ...formats import formats


@dataclass(frozen=True)
class StatsRow:
    name: str
    num_docs: int
    train_pairs: int
    val_pairs: int


class DatasetStats:
    def __init__(self):
        self._rows = []

    def add(self, name, ds):
        self._rows.append(
            StatsRow(
                name=name,
                num_docs=len(ds[formats.KEY_DOCUMENTS]),
                train_pairs=len(ds[formats.KEY_TRAIN_QUERIES]),
                val_pairs=len(ds[formats.KEY_VALIDATION_QUERIES]),
            )
        )

    @staticmethod
    def _fmt_int(n):
        return f"{int(n):,}"

    def to_string(self):
        headers = ("Dataset", "|D|", "Train Pairs", "Val Pairs")

        rendered_rows = []
        for r in self._rows:
            rendered_rows.append(
                (
                    r.name,
                    self._fmt_int(r.num_docs),
                    self._fmt_int(r.train_pairs),
                    self._fmt_int(r.val_pairs),
                )
            )

        w0 = max(len(headers[0]), *(len(row[0]) for row in rendered_rows), 7)
        w1 = max(len(headers[1]), *(len(row[1]) for row in rendered_rows), 3)
        w2 = max(len(headers[2]), *(len(row[2]) for row in rendered_rows), 11)
        w3 = max(len(headers[3]), *(len(row[3]) for row in rendered_rows), 9)

        widths = (w0, w1, w2, w3)

        def hline(left, mid, right):
            return (
                left
                + "─" * (widths[0] + 2)
                + mid
                + "─" * (widths[1] + 2)
                + mid
                + "─" * (widths[2] + 2)
                + mid
                + "─" * (widths[3] + 2)
                + right
            )

        def fmt_header(cols):
            return (
                f"│ {cols[0]:^{widths[0]}} │ {cols[1]:^{widths[1]}} │ {cols[2]:^{widths[2]}} │ {cols[3]:>{widths[3]}} │"
            )

        def fmt_row(cols):
            return (
                f"│ {cols[0]:<{widths[0]}} │ {cols[1]:>{widths[1]}} │ {cols[2]:>{widths[2]}} │ {cols[3]:>{widths[3]}} │"
            )

        lines = []
        lines.append(hline("┌", "┬", "┐"))
        lines.append(fmt_header(headers))
        lines.append(hline("├", "┼", "┤"))

        for row in rendered_rows:
            lines.append(fmt_row(row))

        lines.append(hline("└", "┴", "┘"))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string()


DEFAULT_SPLIT_NAME = "default"


def stats(data, config):
    stats = DatasetStats()

    if isinstance(data, list):
        for ds_split_name, ds_split in data:
            stats.add(ds_split_name, ds_split)
    else:
        stats.add(DEFAULT_SPLIT_NAME, data)

    print(stats)
