# Optional phone integration

Folder-based processing and the synthetic demo work without a phone. The source tree also contains existing phone-capture integration code.

## Companion sensor app

`fc_link.py` looks for Android Debug Bridge (ADB) in a local Android SDK, on `PATH`, or in a bundled folder. Obtain Android Platform Tools from the official Android developer distribution if needed. Bundled ADB executables and DLLs are excluded from this repository.

The sensor workflow expects an Android application with package ID `com.fatiguecam.sensor` and a matching protocol. Its APK exists in the original working environment, but its source and redistribution rights have not been reviewed here. The APK is not included and this workflow is not reproducible from this repository alone.

The existing install action runs ADB installation with permission grants. Use it only for a trusted companion application and an explicitly selected laboratory device. This repository preparation does not install or connect to a phone.

## Google Camera workflow

`gcam_b.py` is tailored to a Pixel 8 Pro Google Camera screen layout. It drives the shutter through ADB, copies completed images to the computer and then attempts to delete the collected images from the phone. It also sends focus/zoom gestures and screen captures for preview. This is device-specific automation and has not been hardware-tested in the repository preparation.

Do not assume it works with arbitrary Android devices or camera UI versions. Use an isolated test device and confirm the workflow before enabling it on a research run.

## Protocol boundary

The local phone link uses loopback TCP, with ADB forwarding in USB mode. It has size limits for incoming JSON/binary frames but no application-level authentication or encryption. The simulator mentioned in the source is not included. Do not expose the socket to an untrusted network.
