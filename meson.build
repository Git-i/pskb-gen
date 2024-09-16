project('pskb-gen', 'cpp', default_options: ['cpp_std=c++20'])
is_linux = build_machine.system() == 'linux'
cpp = meson.get_compiler('cpp')
executable('main', [
    'main.cpp',
    'generator.cpp'
], dependencies: [
    dependency('dxvk-dxgi', required: is_linux),
    dependency('dxvk-d3d11', required: is_linux),
    dependency('argparse'),
    dependency('pistachio_rhi'),
    dependency('rhi_sc'),
    cpp.find_library('ktx')
])