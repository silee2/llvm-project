//== SystemZInstPrinterCommon.h - Common SystemZ InstPrinter funcs *- C++ -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a SystemZ MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZINSTPRINTERCOMMON_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZINSTPRINTERCOMMON_H

#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegister.h"
#include <cstdint>

namespace llvm {

class MCOperand;

class SystemZInstPrinterCommon : public MCInstPrinter {
public:
  SystemZInstPrinterCommon(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                           const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  // Print an address with the given base, displacement and index.
  void printAddress(const MCAsmInfo *MAI, MCRegister Base,
                    const MCOperand &DispMO, MCRegister Index, raw_ostream &O);

  // Print the given operand.
  void printOperand(const MCOperand &MO, const MCAsmInfo *MAI, raw_ostream &O);

  virtual void printFormattedRegName(const MCAsmInfo *MAI, MCRegister Reg,
                                     raw_ostream &O) {}

  // Override MCInstPrinter.
  void printRegName(raw_ostream &O, MCRegister Reg) override;

protected:
  template <unsigned N>
  void printUImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  template <unsigned N>
  void printSImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);

  // Print various types of operand.
  void printOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printOperand(const MCInst *MI, uint64_t /*Address*/, unsigned OpNum,
                    raw_ostream &O) {
    printOperand(MI, OpNum, O);
  }
  void printBDAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printBDXAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printBDLAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printBDRAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printBDVAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printLXAAddrOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU1ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU2ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU3ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU4ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printS8ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU8ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU12ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printS16ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU16ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printS32ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU32ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU48ImmOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printPCRelOperand(const MCInst *MI, uint64_t Address, int OpNum,
                         raw_ostream &O);
  void printPCRelTLSOperand(const MCInst *MI, uint64_t Address, int OpNum,
                            raw_ostream &O);

  // Print the mnemonic for a condition-code mask ("ne", "lh", etc.)
  // This forms part of the instruction name rather than the operand list.
  void printCond4Operand(const MCInst *MI, int OpNum, raw_ostream &O);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZINSTPRINTERCOMMON_H
