import numpy as np
import pytest

from benchmark.benchmark import build_cases, format_buffer_table, main, make_stream, run_buffer_benchmark


@pytest.mark.parametrize("name", list(build_cases()))
def test_loop_and_vectorised_versions_agree(name):
    loop_function, vectorised_function, arguments = build_cases(seed=1)[name]
    loop_result = np.atleast_1d(loop_function(*arguments))
    vectorised_result = np.atleast_1d(vectorised_function(*arguments))
    for a, b in zip(loop_result, vectorised_result):
        np.testing.assert_allclose(a, b, rtol=1e-9)


def test_buffer_benchmark_reports_fewer_retrains_for_bigger_buffers():
    X, y = make_stream(n_samples=1200)
    rows, trainers = run_buffer_benchmark(X, y, buffer_sizes=(200, 600), chunk_size=100)
    assert [row["n_retrains"] for row in rows] == [6, 2]
    assert set(trainers) == {200, 600}
    assert "buffer" in format_buffer_table(rows)


def test_main_runs_both_parts(capsys, tmp_path):
    main(["--repeats", "1", "--warmup", "0", "--samples", "600",
          "--buffer-sizes", "200", "300", "--plot", str(tmp_path / "buf.png")])
    output = capsys.readouterr().out
    assert "Median speedup" in output
    assert "retrains" in output
    assert (tmp_path / "buf.png").exists()
