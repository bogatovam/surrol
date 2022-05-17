from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    #entry_point='surrol.tasks.needle_reach:NeedleReach',
    entry_point='surrol.tasks:NeedleReach',
    max_episode_steps=150,
)

register(
    id='NeedleGrasp-v0',
    #entry_point='surrol.tasks.needle_reach:NeedleReach',
    entry_point='surrol.tasks:NeedleGrasp',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    #entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    entry_point='surrol.tasks:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    #entry_point='surrol.tasks.needle_pick:NeedlePick',
    entry_point='surrol.tasks:NeedlePick',
    max_episode_steps=200,
)

register(
    id='NeedlePickViaGrasp-v0',
    #entry_point='surrol.tasks.needle_pick:NeedlePick',
    entry_point='surrol.tasks:NeedlePickViaGrasp',
    max_episode_steps=50,
)

register(
    id='NeedlePickPointSpecific-v0',
    #entry_point='surrol.tasks.needle_pick:NeedlePick',
    entry_point='surrol.tasks:NeedlePickPointSpecific',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v0',
    #entry_point='surrol.tasks.peg_transfer:PegTransfer',
    entry_point='surrol.tasks:PegTransfer',
    max_episode_steps=300,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    #entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    entry_point='surrol.tasks:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='NeedleRegrasp_custom-v0',
    #entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    entry_point='surrol.tasks:NeedleRegrasp_custom',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    #entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    entry_point='surrol.tasks:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    #entry_point='surrol.tasks.ecm_reach:ECMReach',
    entry_point='surrol.tasks:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    #entry_point='surrol.tasks.ecm_misorient:MisOrient',
    entry_point='surrol.tasks:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    #entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    entry_point='surrol.tasks:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    #entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    entry_point='surrol.tasks:ActiveTrack',
    max_episode_steps=500,
)