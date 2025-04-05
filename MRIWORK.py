import vtk
import nibabel as nib
import numpy as np
import subprocess
import os

DEFAULT_OPACITY = 1
WINDOW_SIZE = (1000, 1000)
BACKGROUND_COLOR = (0.1, 0.1, 0.1)
ANIMATION_STEPS = 60
ANIMATION_DURATION = 600
ZOOM_FACTOR = 1.1
ROTATION_ANGLE = 5
MOVE_FACTOR = 2.0

class VisualizationState:
    def __init__(self):
        self.isShowBrainOnly = False
        self.isShowBrain = True
        self.targetShowBrainOnly = None
        self.targetShowBrain = None

        self.nonBrain = None
        self.brain = None
        self.renderer = None
        self.roiRenderer = None
        self.interactor = None
        self.isAnimating = False
        self.currentStep = 0
        self.roiPoints = []
        self.isRoiActive = False
        self.roiSpheres = []
        self.clips = []
        self.interactorStyle = None

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, state, renderer, roiRenderer):
        self._state = state
        self._renderer = renderer
        self._roiRenderer = roiRenderer
        self._state.interactorStyle = self
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.leftButtonPressed)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.keyPressed)

    def leftButtonPressed(self, obj, event):
        interactor = self.GetInteractor()
        if interactor.GetShiftKey():
            clickPos = interactor.GetEventPosition()
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(clickPos[0], clickPos[1], 0, self._renderer)
            worldPos = picker.GetPickPosition()

            if len(self._state.roiPoints) < 2:
                self._state.roiPoints.append(worldPos)
                self.addRoiPoint(worldPos)

                if len(self._state.roiPoints) == 2:
                    plane = self.createClip(self._state.roiPoints[0], self._state.roiPoints[1])
                    if plane:
                        self._state.clips.append(plane)
                        self._state.isRoiActive = True
                    self._state.roiPoints = []
                    self.deleteAllRoiPoints()

                self._state.renderer.GetRenderWindow().Render()
        else:
            super().OnLeftButtonDown()

    def deleteRoi(self):
        modified = False
        if self._state.clips:
            self.deleteAllClips()
            self._state.clips = []
            self._state.isRoiActive = False
            modified = True
        if self._state.roiPoints:
            self.deleteLastRoiPoint()
            self._state.roiPoints = []
            modified = True
        return modified

    def deleteLastRoiPoint(self):
        if len(self._state.roiPoints) == 1 and self._state.roiSpheres:
            self._state.roiPoints.pop()
            roiPoint = self._state.roiSpheres.pop()
            self._roiRenderer.RemoveActor(roiPoint)
            return True
        return False

    def deleteLastClip(self):
        if self._state.clips:
            self._state.clips.pop()
            self.applyActualClips()
            if not self._state.clips:
                self._state.isRoiActive = False
            return True
        return False

    def applyActualClips(self):
        if self._state.brain:
            brainMapper = self._state.brain.GetMapper()
            brainMapper.RemoveAllClippingPlanes()
        if self._state.nonBrain:
            nonBrainMapper = self._state.nonBrain.GetMapper()
            nonBrainMapper.RemoveAllClippingPlanes()

        for plane in self._state.clips:
            if self._state.brain:
                self._state.brain.GetMapper().AddClippingPlane(plane)
            if self._state.nonBrain:
                self._state.nonBrain.GetMapper().AddClippingPlane(plane)

    def keyPressed(self, obj, event):
        interactor = self.GetInteractor()
        key = interactor.GetKeySym()
        lowerKey = key.lower()
        camera = self._renderer.GetActiveCamera()
        modified = False

        isCtrlPressed = interactor.GetControlKey()

        if lowerKey == "c":
            modified = self.deleteRoi()
        elif lowerKey == "t":
            startAnimation(self._state, "nonBrain")
        elif lowerKey == "b":
            startAnimation(self._state, "brain")
        elif isCtrlPressed and lowerKey == "z":
            modified = self.deleteLastClip()
        elif lowerKey == "u":
            modified = self.deleteLastRoiPoint()
        elif key == "plus" or key == "equal":
            camera.Dolly(ZOOM_FACTOR)
            modified = True
        elif key == "minus":
            camera.Dolly(1.0 / ZOOM_FACTOR)
            modified = True
        elif key == "Up":
            camera.Elevation(ROTATION_ANGLE)
            modified = True
        elif key == "Down":
            camera.Elevation(-ROTATION_ANGLE)
            modified = True
        elif key == "Left":
            camera.Azimuth(-ROTATION_ANGLE)
            modified = True
        elif key == "Right":
            camera.Azimuth(ROTATION_ANGLE)
            modified = True
        elif lowerKey == 'w':
            viewUp = list(camera.GetViewUp())
            vtk.vtkMath.Normalize(viewUp)
            delta = [MOVE_FACTOR * v for v in viewUp]
            self.moveCamera(camera, delta)
            modified = True
        elif lowerKey == 's':
            viewUp = list(camera.GetViewUp())
            vtk.vtkMath.Normalize(viewUp)
            delta = [-MOVE_FACTOR * v for v in viewUp]
            self.moveCamera(camera, delta)
            modified = True
        elif lowerKey == 'a':
            viewUp = camera.GetViewUp()
            viewPlaneNormal = camera.GetDirectionOfProjection()
            viewRight = [0.0, 0.0, 0.0]
            vtk.vtkMath.Cross(viewPlaneNormal, viewUp, viewRight)
            vtk.vtkMath.Normalize(viewRight)
            delta = [-MOVE_FACTOR * v for v in viewRight]
            self.moveCamera(camera, delta)
            modified = True
        elif lowerKey == 'd':
            viewUp = camera.GetViewUp()
            viewPlaneNormal = camera.GetDirectionOfProjection()
            viewRight = [0.0, 0.0, 0.0]
            vtk.vtkMath.Cross(viewPlaneNormal, viewUp, viewRight)
            vtk.vtkMath.Normalize(viewRight)
            delta = [MOVE_FACTOR * v for v in viewRight]
            self.moveCamera(camera, delta)
            modified = True
        elif lowerKey == 'z' and not isCtrlPressed:
            camera.Roll(ROTATION_ANGLE)
            modified = True
        elif lowerKey == 'x':
            camera.Roll(-ROTATION_ANGLE)
            modified = True
        else:
            super().OnKeyPress()

        if modified:
            self._renderer.ResetCameraClippingRange()
            self._state.renderer.GetRenderWindow().Render()

    def moveCamera(self, camera, delta):
        currentPos = list(camera.GetPosition())
        currentFp = list(camera.GetFocalPoint())
        newPos = [currentPos[i] + delta[i] for i in range(3)]
        newFp = [currentFp[i] + delta[i] for i in range(3)]
        camera.SetPosition(newPos)
        camera.SetFocalPoint(newFp)

    def addRoiPoint(self, point):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(point)
        sphereSource.SetRadius(1.0)
        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
        sphere = vtk.vtkActor()
        sphere.SetMapper(sphereMapper)
        sphere.GetProperty().SetColor(1, 0, 0)
        sphere.GetProperty().SetAmbient(1.0)
        sphere.GetProperty().SetDiffuse(0.0)
        sphere.GetProperty().SetSpecular(0.0)
        sphereMapper.SetResolveCoincidentTopologyToPolygonOffset()
        self._roiRenderer.AddActor(sphere)
        self._state.roiSpheres.append(sphere)

    def deleteAllRoiPoints(self):
        while self._state.roiSpheres:
            actor = self._state.roiSpheres.pop()
            self._roiRenderer.RemoveActor(actor)

    def createClip(self, point1, point2):
        lineVector = [point2[i] - point1[i] for i in range(3)]
        
        if vtk.vtkMath.Normalize(lineVector) == 0:
            return None

        camera = self._renderer.GetActiveCamera()
        cameraPos = camera.GetPosition()
        camToPoint1 = [point1[i] - cameraPos[i] for i in range(3)]

        normal = [0, 0, 0]
        vtk.vtkMath.Cross(lineVector, camToPoint1, normal)
        
        if vtk.vtkMath.Normalize(normal) == 0:
            viewPlaneNormal = camera.GetDirectionOfProjection()
            vtk.vtkMath.Cross(lineVector, viewPlaneNormal, normal)
            if vtk.vtkMath.Normalize(normal) == 0:
                return None

        plane = vtk.vtkPlane()
        plane.SetOrigin(point1)
        plane.SetNormal(normal)

        if self._state.brain:
            self._state.brain.GetMapper().AddClippingPlane(plane)
        if self._state.nonBrain:
            self._state.nonBrain.GetMapper().AddClippingPlane(plane)

        return plane

    def deleteAllClips(self):
        if self._state.brain:
            self._state.brain.GetMapper().RemoveAllClippingPlanes()
        if self._state.nonBrain:
            self._state.nonBrain.GetMapper().RemoveAllClippingPlanes()

def startAnimation(state, targetActorName):
    if state.isAnimating:
        return

    state.isAnimating = True
    state.currentStep = 0

    if targetActorName == "nonBrain":
        state.targetShowBrainOnly = not state.isShowBrainOnly
    elif targetActorName == "brain":
        state.targetShowBrain = not state.isShowBrain

    state.interactor.AddObserver("TimerEvent", lambda obj, event: animateTransition(obj, event, state))
    state.interactor.CreateRepeatingTimer(int(ANIMATION_DURATION / ANIMATION_STEPS))

def animateTransition(obj, event, state):
    if not state.isAnimating:
        return

    progress = state.currentStep / ANIMATION_STEPS

    if state.targetShowBrainOnly is not None:
        nonBrainOpacity = 1.0 - progress if state.targetShowBrainOnly else progress
        updateOpacity(state.nonBrain, nonBrainOpacity)

    if state.targetShowBrain is not None:
        brainOpacity = progress if state.targetShowBrain else 1.0 - progress
        updateOpacity(state.brain, brainOpacity)

    state.renderer.GetRenderWindow().Render()
    state.currentStep += 1

    if state.currentStep > ANIMATION_STEPS:
        state.isAnimating = False
        if state.targetShowBrainOnly is not None:
            state.isShowBrainOnly = state.targetShowBrainOnly
            state.targetShowBrainOnly = None
        if state.targetShowBrain is not None:
            state.isShowBrain = state.targetShowBrain
            state.targetShowBrain = None
        state.currentStep = 0
        state.interactor.DestroyTimer()

def updateOpacity(actor, opacity):
    if actor:
        volumeProperty = actor.GetProperty()
        opacityTf = volumeProperty.GetScalarOpacity()
        opacityTf.RemoveAllPoints()
        minVal, maxVal = 0, 255
        opacityTf.AddPoint(minVal, 0.0)
        opacityTf.AddPoint(minVal + 0.1 * (maxVal - minVal), 0.0)
        opacityTf.AddPoint(minVal + 0.5 * (maxVal - minVal), opacity * 0.5)
        opacityTf.AddPoint(maxVal, opacity)

def createVolume(data, affine, opacity, isBrain):
    importer = vtk.vtkImageImport()
    dataBytes = data.tobytes()
    importer.CopyImportVoidPointer(dataBytes, len(dataBytes))
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)

    shape = data.shape
    importer.SetDataExtent(0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)
    importer.SetWholeExtent(0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)

    spacing = nib.affines.voxel_sizes(affine)
    origin = affine[:3, 3]
    importer.SetDataSpacing(*spacing)
    importer.SetDataOrigin(origin)
    importer.Update()

    # volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputConnection(importer.GetOutputPort())
    volumeMapper.SetBlendModeToComposite()

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.4)
    volumeProperty.SetDiffuse(0.6)
    volumeProperty.SetSpecular(0.2)

    minVal, maxVal = 0, 255
    opacityTf = vtk.vtkPiecewiseFunction()
    colorTf = vtk.vtkColorTransferFunction()

    opacityTf.AddPoint(minVal, 0.0)
    opacityTf.AddPoint(minVal + 0.1 * (maxVal - minVal), 0.0)
    opacityTf.AddPoint(minVal + 0.5 * (maxVal - minVal), opacity * 0.5)
    opacityTf.AddPoint(maxVal, opacity)

    if isBrain:
        colorTf.AddRGBPoint(minVal, 0.5, 0.3, 0.3)
        colorTf.AddRGBPoint(minVal + 0.2 * (maxVal - minVal), 0.7, 0.5, 0.5)
        colorTf.AddRGBPoint(minVal + 0.6 * (maxVal - minVal), 0.9, 0.7, 0.7)
        colorTf.AddRGBPoint(maxVal, 1.0, 0.8, 0.8)
    else:
        colorTf.AddRGBPoint(minVal, 0.85, 0.75, 0.7)
        colorTf.AddRGBPoint(minVal + 0.2 * (maxVal - minVal), 0.9, 0.8, 0.75)
        colorTf.AddRGBPoint(minVal + 0.6 * (maxVal - minVal), 0.95, 0.85, 0.8)
        colorTf.AddRGBPoint(maxVal, 1.0, 0.9, 0.85)

    volumeProperty.SetScalarOpacity(opacityTf)
    volumeProperty.SetColor(colorTf)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    return volume

def runFsl(inputFilePath, outputFilePath, betParams):
    betCommand = ['bet', inputFilePath, outputFilePath] + betParams
    subprocess.run(betCommand, check=True, env=os.environ.copy())
    return outputFilePath

def loadNifti(filePath):
    img = nib.as_closest_canonical(nib.load(filePath))
    data = img.get_fdata()
    affine = img.affine
    return data, affine, img

def normalizeData(data, usePercentiles=True):
    if usePercentiles:
        p1, p99 = np.percentile(data[data > 0], [1, 99])
    else:
        p1, p99 = data.min(), data.max()
    dataClipped = np.clip(data, p1, p99)
    dataNorm = ((dataClipped - p1) / (p99 - p1 + 1e-6) * 255).astype(np.uint8)
    return dataNorm

def createButton(bounds, textLabel, fontSize=14, opacity=0.8, bgColor=(100,100,100), textColor=(1,1,1)):
    xMin, xMax, yMin, yMax = bounds

    buttonBackground = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    points.InsertNextPoint(xMin, yMin, 0)
    points.InsertNextPoint(xMax, yMin, 0)
    points.InsertNextPoint(xMax, yMax, 0)
    points.InsertNextPoint(xMin, yMax, 0)

    quad = vtk.vtkPolygon()
    quad.GetPointIds().SetNumberOfIds(4)
    for i in range(4):
        quad.GetPointIds().SetId(i, i)
    polys.InsertNextCell(quad)

    for _ in range(4):
        colors.InsertNextTypedTuple(bgColor)

    buttonBackground.SetPoints(points)
    buttonBackground.SetPolys(polys)
    buttonBackground.GetPointData().SetScalars(colors)

    buttonMapper = vtk.vtkPolyDataMapper2D()
    buttonMapper.SetInputData(buttonBackground)

    button = vtk.vtkActor2D()
    button.SetMapper(buttonMapper)
    button.GetProperty().SetOpacity(opacity)

    buttonText = vtk.vtkTextActor()
    buttonText.SetInput(textLabel)
    textProp = buttonText.GetTextProperty()
    textProp.SetFontSize(fontSize)
    textProp.SetColor(*textColor)
    textProp.SetJustificationToCentered()
    textProp.SetVerticalJustificationToCentered()

    buttonText.GetPositionCoordinate().SetCoordinateSystemToDisplay()
    buttonText.GetPositionCoordinate().SetValue((xMin + xMax) / 2, (yMin + yMax) / 2)

    return button, buttonText

def setupUi(renderer, interactor, state):
    buttonSpace = 10
    buttonHeight = 40
    toggleButtonWidth = 120
    resetButtonWidth = 80

    toggleBounds = (
        buttonSpace,
        buttonSpace + toggleButtonWidth,
        buttonSpace,
        buttonSpace + buttonHeight
    )

    brainBounds = (
        toggleBounds[1] + buttonSpace,
        toggleBounds[1] + buttonSpace + toggleButtonWidth,
        buttonSpace,
        buttonSpace + buttonHeight
    )

    resetBounds = (
        brainBounds[1] + buttonSpace,
        brainBounds[1] + buttonSpace + resetButtonWidth,
        buttonSpace,
        buttonSpace + buttonHeight
    )

    toggleButton, toggleButtonText = createButton(toggleBounds, "Toggle View")
    brainButton, brainButtonText = createButton(brainBounds, "Toggle Brain")
    resetButton, resetButtonText = createButton(resetBounds, "Reset")

    renderer.AddActor2D(toggleButton)
    renderer.AddActor2D(toggleButtonText)
    renderer.AddActor2D(brainButton)
    renderer.AddActor2D(brainButtonText)
    renderer.AddActor2D(resetButton)
    renderer.AddActor2D(resetButtonText)

    def buttonPressed(obj, event):
        if event == "LeftButtonPressEvent":
            interactor = obj
            if not interactor.GetShiftKey():
                clickPos = interactor.GetEventPosition()
                x, y = clickPos[0], clickPos[1]

                txMin, txMax, tyMin, tyMax = toggleBounds
                if txMin <= x <= txMax and tyMin <= y <= tyMax:
                    startAnimation(state, "nonBrain")
                    return

                bxMin, bxMax, byMin, byMax = brainBounds
                if bxMin <= x <= bxMax and byMin <= y <= byMax:
                    startAnimation(state, "brain")
                    return

                rxMin, rxMax, ryMin, ryMax = resetBounds
                if rxMin <= x <= rxMax and ryMin <= y <= ryMax:
                    if state.interactorStyle and state.interactorStyle.deleteRoi():
                        state.renderer.GetRenderWindow().Render()
                    return

    interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, buttonPressed, 1.0)

def setupInfoText(img, brainData, filePath, renderer):
    voxelSizes = nib.affines.voxel_sizes(img.affine)
    voxelSizes = tuple(round(float(x), 2) for x in voxelSizes)
    brainVoxelCount = np.sum(brainData > 0)
    voxelVolume = np.prod(voxelSizes)
    brainVolume = brainVoxelCount * voxelVolume

    infoTextFormat = (
        f"File: {os.path.basename(filePath)}\n"
        f"Dimensions (x, y, z): {img.shape}\n"
        f"Slices (z): {img.shape[2]}\n"
        f"Voxel Size (mm): {voxelSizes}\n"
        f"Slice Spacing (mm): {round(float(voxelSizes[2]), 2)}\n"
        f"Orientation: {''.join(nib.orientations.aff2axcodes(img.affine))}\n"
        f"Brain Volume (cm³): {brainVolume/1000:.2f}"
    )

    infoText = vtk.vtkTextActor()
    infoText.SetInput(infoTextFormat)
    textProp = infoText.GetTextProperty()
    textProp.SetFontSize(16)
    textProp.SetColor(1, 1, 1)
    textProp.SetJustificationToLeft()
    textProp.SetVerticalJustificationToTop()

    infoText.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    infoText.GetPositionCoordinate().SetValue(0.01, 0.99)

    renderer.AddActor2D(infoText)
    return infoText

def setupVisualization(originalData, brainData, affine, originalImg, brainImgData, inputFilePath):
    state = VisualizationState()

    renderer = vtk.vtkRenderer()
    renderer.SetLayer(0)
    renderer.SetInteractive(1)

    roiRenderer = vtk.vtkRenderer()
    roiRenderer.SetLayer(1)
    roiRenderer.SetInteractive(0)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetNumberOfLayers(2)
    renderWindow.AddRenderer(renderer)
    renderWindow.AddRenderer(roiRenderer)
    renderWindow.SetSize(*WINDOW_SIZE)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    interactorStyle = CustomInteractorStyle(state, renderer, roiRenderer)
    interactor.SetInteractorStyle(interactorStyle)
    interactor.SetDesiredUpdateRate(30)

    state.renderer = renderer
    state.roiRenderer = roiRenderer
    state.interactor = interactor

    roiRenderer.SetActiveCamera(renderer.GetActiveCamera())

    brainMask = brainData > 0
    originalData[brainMask] = 0

    brain = createVolume(brainData, affine, DEFAULT_OPACITY, True)
    nonBrain = createVolume(originalData, affine, DEFAULT_OPACITY, False)

    state.brain = brain
    state.nonBrain = nonBrain

    renderer.AddVolume(nonBrain)
    renderer.AddVolume(brain)

    setupUi(renderer, interactor, state)
    setupInfoText(originalImg, brainImgData, inputFilePath, renderer)

    renderer.ResetCamera()
    renderer.SetBackground(*BACKGROUND_COLOR)
    roiRenderer.SetBackground(0, 0, 0)
    roiRenderer.SetBackgroundAlpha(0.0)

    renderWindow.Render()
    interactor.Initialize()
    interactor.Start()

def main():
    inputFilePath = "/home/inda/coding/3dmri/brainSet/brain1.nii"
    betParameters = ['-f', '0.3']

    if not os.path.exists(inputFilePath):
        print(f"No file: {inputFilePath}")
        return

    baseName = os.path.basename(inputFilePath)
    dirName = os.path.dirname(inputFilePath)
    fileStem = baseName.replace(".nii.gz", "").replace(".nii", "")
    outputPath = os.path.join(dirName, f"{fileStem}_brain.nii.gz")

    if not os.path.exists(outputPath):
        runFsl(inputFilePath, outputPath, betParameters)

    originalData, affine, originalImg = loadNifti(inputFilePath)

    brainData, _, brainImg = loadNifti(outputPath)
    brainImgData = brainImg.get_fdata()

    originalDataNorm = normalizeData(originalData)
    brainDataNorm = normalizeData(brainData)

    originalDataNorm = np.transpose(originalDataNorm, (1, 2, 0)).copy()
    brainDataNorm = np.transpose(brainDataNorm, (1, 2, 0)).copy()

    setupVisualization(originalDataNorm, brainDataNorm, affine, originalImg, brainImgData, inputFilePath)

if __name__ == "__main__":
    main()