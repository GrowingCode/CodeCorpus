package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.swing.JOptionPane;

import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;

public class ImportAlreadyDocument {

	public static void main(String[] args) {
		try {
			PdfReader reader = new PdfReader("c:\\newPdfFile.pdf");// 创建已有文档的PdfReader对象
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
					"c:\\导入已有文档.pdf"));// 关联已有文档与输出流
			stamp.close();// 关闭PdfStamper对象，完成文档导入功能
			JOptionPane.showMessageDialog(null, "导入成功...\n完成已有文档的导入。");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
