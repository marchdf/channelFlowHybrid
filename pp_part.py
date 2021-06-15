# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import numpy as np
import scipy.spatial.qhull as qhull
import pandas as pd
from mpi4py import MPI
import stk
import utilities


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="An post-processing tool for a sideset"
    )
    parser.add_argument(
        "-m", "--mfile", help="Root name of files to postprocess", required=True
    )
    parser.add_argument("--auto_decomp", help="Auto-decomposition", action="store_true")
    parser.add_argument("-p", "--part", help="Part to post-process", required=True)
    args = parser.parse_args()

    fdir = os.path.dirname(args.mfile)
    pfx = os.path.splitext(os.path.basename(args.mfile))[0]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    par = stk.Parallel.initialize()
    printer = utilities.p0_printer(par)

    mesh = stk.StkMesh(par)
    printer("Reading meta data for mesh: ", args.mfile)
    mesh.read_mesh_meta_data(args.mfile, auto_decomp=args.auto_decomp)
    printer("Done reading meta data")

    printer("Loading bulk data for mesh: ", args.mfile)
    mesh.populate_bulk_data()
    printer("Done reading bulk data")

    num_time_steps = mesh.stkio.num_time_steps
    max_time = mesh.stkio.max_time
    tsteps = mesh.stkio.time_steps
    printer(f"""Num. time steps = {num_time_steps}\nMax. time step  = {max_time}""")

    idf = pd.DataFrame(
        {
            "t": tsteps,
            "u": np.zeros(len(tsteps)),
            "tke": np.zeros(len(tsteps)),
            "sdr": np.zeros(len(tsteps)),
        }
    )
    for k, tstep in enumerate(tsteps):
        ftime, missing = mesh.stkio.read_defined_input_fields(tstep)
        printer(f"Loaded fields for time: {ftime}")

        coords = mesh.meta.coordinate_field

        # Extract fields at the parts
        m_part = mesh.meta.get_part(args.part)
        sel = m_part & mesh.meta.locally_owned_part
        fields = [
            mesh.meta.get_field("velocity"),
            mesh.meta.get_field("turbulent_ke"),
            mesh.meta.get_field("specific_dissipation_rate"),
        ]
        names = ["x", "y", "z", "u", "v", "w", "tke", "sdr"]

        optional_fields = {"beta": "k_ratio", "rk": "avg_res_adequacy_parameter"}
        for key, value in optional_fields.items():
            fld = mesh.meta.get_field(value)
            if not fld.is_null:
                fields.append(fld)
                names.append(key)

        cnt = 0
        nnodes = sum(bkt.size for bkt in mesh.iter_buckets(sel, stk.StkRank.NODE_RANK))
        data = np.zeros((nnodes, len(names)))
        for bkt in mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            arr = coords.bkt_view(bkt)
            for fld in fields:
                vals = fld.bkt_view(bkt)
                if len(vals.shape) == 1:  # its a scalar
                    vals = vals.reshape(-1, 1)
                arr = np.hstack((arr, vals))
            data[cnt : cnt + bkt.size, :] = arr
            cnt += bkt.size

        lst = comm.gather(data, root=0)
        comm.Barrier()
        if rank == 0:
            df = pd.DataFrame(np.vstack(lst), columns=names)
            if tstep == tsteps[-1]:
                df.to_csv(os.path.join(fdir, f"{pfx}-f_{args.part}.dat"), index=False)
            means = df.groupby("y", as_index=False).mean().sort_values(by=["y"])
            Ly = df.y.max() - df.y.min()
            idf.iloc[k].u = np.trapz(means.u, means.y) / Ly
            idf.iloc[k].tke = np.trapz(means.tke, means.y) / Ly
            idf.iloc[k].sdr = np.trapz(means.sdr, means.y) / Ly

    if rank == 0:
        idf.to_csv(os.path.join(fdir, f"{pfx}-{args.part}.dat"), index=False)
